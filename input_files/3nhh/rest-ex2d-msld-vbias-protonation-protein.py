"""
Run MSLD simulations with charge & fixed bias replica exchange under NVT conditions 
for titratable residues in a protein which contains a salt bridge.
PME is used for electrostatic interactions.
BLaDE-GPU is used to acclerate MD simulations.
For variable biases, we use quadratic term, end-point term and soft term, 
even though we don't use soft-core here.
Use Ryan's latest restraint SCAT to hold analogous atoms together.
2D replica exchange (partial charge of the flexible region and fixed bias) is used.
No pH-dependent term is used.
"""

from mpi4py import MPI

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import shlex
import json
from scipy.io import FortranFile

import pycharmm
import pycharmm.read as read
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.settings as settings
import pycharmm.write as write
import pycharmm.nbonds as nbonds
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.select as select
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.param as param
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.shake as shake
import pycharmm.scalar as scalar
import pycharmm.charmm_file as charmm_file

# platform
platform='blade'
# soft-core potential
soft=False
# number of digits after the decimal point
ndec=4
debug=False

comp=sys.argv[1]
itt=int(sys.argv[2])
gpus_per_node=int(sys.argv[3])
ns='3'

# replica exchange setup 
nbias=6  # number of V_bias windows
nchrg=4  # number of charge windows
tsim=298.15
tmax=330
tmin=tsim

comm=MPI.COMM_WORLD
nproc=comm.Get_size()
rank=comm.Get_rank()

cuda_device_id=rank%gpus_per_node
print('cuda_device_id is',cuda_device_id)

pka={'ASP':4.0,
     'GLU':4.4,
     'HSD':7.0,
     'HSE':6.6,
     'LYS':10.40}

def lambda_schedule(n_chrg,delta_fixed=None,n_bias=15,linear_chrg=False,geometric_chrg=True,sin_chrg=False):
    """
    get lambda schedule based on number of charge and V_bias windows

    Parameters
    ----------
    n_chrg : int
        number of charge windows
    n_bias : int
        number of V_bias windows
    linear_chrg : bool
        wheather charge windows are linear
    geometric_chrg : bool
        wheather charge windows are geometri series
    sin_chrg: bool
        wheather charge windows follow a sin funtion

    Returns
    ------- 
    cond_array : numpy.array
        a 2D array of with (V_bias, lambda_chrg) for each condition,
        where lambda_chrg is the scaling factor for charges in the flexible region,
        V_bias is the SHIFT of fixed biasing potential
    """
    if geometric_chrg==True and linear_chrg==False and sin_chrg==False:
       # effective temp
       temp=tmin*np.exp(np.arange(n_chrg)*np.log(tmax/tmin)/(n_chrg-1))
       # 1D array with a shape of (n_chrg, 1): in each row, we have lambda of the flexible region
       # keep only 4 digits after the decimal point
       chrg_array=np.array(tsim/temp).round(decimals=ndec)
    elif linear_chrg==True and geometric_chrg==False and sin_chrg==False:
       chrg_array=(1-np.arange(n_chrg)*(1-tmin/tmax)/(n_chrg-1)).round(decimals=ndec)
    elif sin_chrg==True and geometric_chrg==False and geometric_chrg==False:
       chrg_array=(np.cos(np.pi/2*np.arange(n_chrg)/(n_chrg-1))*(1-tmin/tmax)+tmin/tmax).round(decimals=ndec)
    else:
       print("ERROR! Don't know how to set up charge scaling factors")
       quit()

    # shift of V_bias (fixed term only), linearly spaced
    if delta_fixed==None:
       delta_fixed=np.log(10)*kB*tsim
    bias_array=np.arange(-n_bias/2,n_bias/2)*delta_fixed

    cond_array=np.round(np.array(np.meshgrid(bias_array,chrg_array)).T.reshape(-1,2),ndec)

    # save the condition file
    if rank==0:
       np.savetxt(cnd_fn,cond_array,fmt='%.4f')

    return cond_array 

def distributeWork(n):
    """determine work per node based on nproc and nodes
    input:   n <- number of processes being employed in calculation
    output:  work[0:n] is a dictionary that contains number of passes
             of calculation done by process rank
    """
    work = {}     # this dictionary contains the work per process
    work[rank] = []
    tasks = list(range(n))
    while len(tasks) > 0:
        if rank < len(tasks):
            work[rank].append(tasks[rank])
        del tasks[:nproc]
    return work

def setup_dir():
    os.system('mkdir -p '+out_dir+'/dcd')
    os.system('mkdir -p '+out_dir+'/res')
    os.system('mkdir -p '+out_dir+'/pdb')
    os.system('mkdir -p '+out_dir+'/out')
    os.system('mkdir -p '+out_dir+'/exc')
    os.system('mkdir -p '+out_dir+'/his')

name=comp
kB=0.0019872042586408316 # kcal/mol/K
beta=1/kB/tsim
segid='PRO0'

ncycles=500         # 1 ns
nsteps_per_cyc=1000 # 2 ps per cycle 
ittm1=itt-1

prm_dir='../../toppar_c36_jul21'
wrk_dir='.'

inp_psf_fn=f'{wrk_dir}/hybrid-solv.psf'
ini_pdb_fn=f'{wrk_dir}/hybrid-solv.pdb'
inp_box_fn=f'{wrk_dir}/box.dat'
cnd_fn=wrk_dir+'/cond.dat'
var_fn=wrk_dir+'/variables/var.txt'
titr_res_fn=f'{wrk_dir}/{comp}-titr-res.dat'

dcd_unit=40
rst_unit=50
rpr_unit=70  # unit previous restart file
if platform=='blade':
   lmd_unit=80  # file unit of lambda file


def open_clog(log_fn):
    '''
    specify charmm output file for a given system
    '''
    clog=charmm_file.CharmmFile(file_name=log_fn,read_only=False,formatted=True)
    lingo.charmm_script('outu '+str(clog.file_unit))
    return clog

def close_clog(clog):
    lingo.charmm_script('outu 6')
    clog.close()

def read_param():
    read.rtf(prm_dir+'/top_all36_prot_hedi_xrliu.rtf')
    read.prm(prm_dir+'/par_all36m_prot.prm',flex=True)
    lingo.charmm_script('stream '+prm_dir+'/toppar_water_ions.str')

def read_init():
    read.psf_card(inp_psf_fn)
    read.pdb(ini_pdb_fn,resid=True)
    lingo.charmm_script('coor stat')
    define_sub(titr_res,radius=5,flex_all=True)
    write.coor_pdb('{}/flex.pdb'.format(out_dir),sele='flex end')

    # get original partial charges
    chrg_df_orig=pd.DataFrame(scalar.get_charges(),columns=['chrg'])

    # using wmain to get which atoms belong to the flex region
    n = psf.get_natom()
    wmain0=coor.get_weights()
    coor.set_weights([0.0] * n)
    lingo.charmm_script('scalar wmain set 1.0 sele flex end')
    wmain1=coor.get_weights()
    inx_flex=pd.DataFrame(wmain1,columns=['wmain'])['wmain']==1.0
    inx_flex.to_csv(f'{out_dir}/inx_flex.dat')
    coor.set_weights(wmain0)

    return chrg_df_orig, inx_flex

def titr_grp(resn):
    '''
    atom names in the titratable group for a given amino acid
    '''
    if resn == 'ASP':
       type_list=['CB','HB1','HB2','CG','OD1','OD2','HD1','HD2']
    elif resn == 'GLU':
       type_list=['CG','HG1','HG2','CD','OE1','OE2','HE1','HE2']
    elif resn == 'HSP' or resn == 'HSD' or resn == 'HSE':
       type_list=['CB','HB1','HB2','CG','ND1','HD1','CE1','HE1','CD2','HD2','NE2','HE2']
    elif resn == 'LYS':
       type_list=['CE','HE1','HE2','NZ','HZ1','HZ2','HZ3']
    return type_list

def define_sub(titr_res_dict,radius=5,flex_all=False,resid_flex=''):
    '''
    define substituents and flexible region
    '''
    sele_flex=~pycharmm.SelectAtoms(select_all=True)
    for aa in titr_res_dict.keys():
        resn=aa.upper()
        atom_list=titr_grp(resn)
        for ires in titr_res_dict[resn][0]:
            resid=str(ires)
            if flex_all==True:
               sele_flex = sele_flex | ( pycharmm.SelectAtoms(seg_id=segid) & pycharmm.SelectAtoms(res_id=resid) )
            w_str=' '.join([i+'W' for i in atom_list])
            m_str=' '.join([i+'M' for i in atom_list])
            p_str=' '.join([i+'P' for i in atom_list])
            u_str=' '.join([i+'U' for i in atom_list])
            sele_w=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=resid,atom_type=w_str)
            sele_m=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=resid,atom_type=m_str)
            sele_p=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=resid,atom_type=p_str)
            sele_u=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=resid,atom_type=u_str)
            sele_w.store(f'site{resn}{resid}subW')
            sele_m.store(f'site{resn}{resid}subM')
            sele_p.store(f'site{resn}{resid}subP')
            sele_u.store(f'site{resn}{resid}subU')
            if debug:
               lingo.charmm_script(f'print coor sele site{resn}{resid}subW end')
               lingo.charmm_script(f'print coor sele site{resn}{resid}subM end')
               lingo.charmm_script(f'print coor sele site{resn}{resid}subP end')
               lingo.charmm_script(f'print coor sele site{resn}{resid}subU end')
    if flex_all==False:
       sele_flex = sele_flex | ( pycharmm.SelectAtoms(seg_id=segid) & pycharmm.SelectAtoms(res_id=resid_flex) )
    sele_flex.store(f'flex')
    if debug:
       lingo.charmm_script(f'print coor sele flex end')


def read_var(fn_var):
    return pd.read_csv(fn_var,index_col=0,header=0)

class setup_block:
    '''
    Set up block for MSLD simulation using charge REST.
    Note that the flexible region doesn't have its own block.
    Later we just reset partial charges of the flexible region.
    '''
    def __init__(self,bias,titr_res_dict,V_bias,lmds=None):
        self.Gbias=bias
        self.titr_res_dict=titr_res_dict
        self.V_bias=V_bias
        self.lmds=lmds
    def run(self):
        self.reset()
        self.call()
    def update_vbias(self,lmd):
        block_str=f'BLOCK\n'
        ldin_str='      !   id lmd vel mass energy friction-coefficient\n'
        ldin_str+='      ldin 1 1.0000  0.0  12.0     0.00  5.0\n'

        iblock=2
        isite=0
        for aa in self.titr_res_dict.keys():
            resn=aa.upper()
            for ires in self.titr_res_dict[resn][0]:
                resid=str(ires)
                utag=self.set_utag(resn)
                refpka=self.set_pka(resn)
                sele_name_lst=self.set_name(resn,resid)
                for isub in range(nsubs[isite]):
                    #print(isite,isub)
                    sele_name=sele_name_lst[isub]
                    if isub==0:
                        phtag='NONE'
                        V_fixed_shift=0
                    else:
                        phtag=utag+' '+str(refpka[isub-1])
                        V_fixed_shift=self.V_bias
                    ldin_str+='      ldin {} {:19.17f}  0.0  12.0  {:7.2f}  5.0\n'.format(iblock,lmd[iblock-2],self.Gbias.loc[f'lams{isite+1}s{isub+1}','value']+V_fixed_shift)
                    iblock+=1
                isite+=1
        block_str+=ldin_str
        block_str+=f'END\n'
        lingo.charmm_script(block_str)
    def set_utag(self,resname):
        if resname == 'ASP' or resname == 'GLU':
           utag='UNEG'
        elif resname == 'HSD' or resname == 'HSE' or resname == 'HSP' or resname == 'LYS':
           utag='UPOS'
        return utag
    def set_name(self,resname,res_id):
        sele_name_w='site%s%ssubW'%(resname,res_id)
        sele_name_m='site%s%ssubM'%(resname,res_id)
        sele_name_p='site%s%ssubP'%(resname,res_id)
        sele_name_u='site%s%ssubU'%(resname,res_id)
        if resname == 'ASP' or resname == 'GLU':
           sele_name=[sele_name_w,sele_name_m,sele_name_p]
        elif resname == 'HSD':
           sele_name=[sele_name_w,sele_name_u]
        elif resname == 'HSE':
           sele_name=[sele_name_w,sele_name_m]
        elif resname == 'LYS':
           sele_name=[sele_name_w,sele_name_m]
        elif resname == 'HSP':
           sele_name=[sele_name_w,sele_name_u,sele_name_m]
        return sele_name
    def set_pka(self,resname):
        if resname == 'HSP':
           refpka_list=[pka['HSD'],pka['HSE']]
        else:
           refpka_list=[pka[resname],pka[resname]]
        return refpka_list
    def reset(self):
        lingo.charmm_script('''
              BLOCK {}
                    clear 
              END
        '''.format(nblocks+1))
    def call(self):
        '''
        don't put these commands into multiple block sections.
        otherwise energy can be incorrect in domdec and blade.
        '''

        block_str=f'BLOCK {nblocks+1}\n'
        ldin_str='      !   id lmd vel mass energy friction-coefficient\n'
        ldin_str+='      ldin 1 1.0000  0.0  12.0     0.00  5.0\n'
        msld_str='      msld 0 '

        iblock=2
        isite=0
        for aa in self.titr_res_dict.keys():
            resn=aa.upper()
            for ires in self.titr_res_dict[resn][0]:
                resid=str(ires)
                utag=self.set_utag(resn)
                refpka=self.set_pka(resn)
                sele_name_lst=self.set_name(resn,resid)
                for isub in range(nsubs[isite]):
                    #print(isite,isub)
                    sele_name=sele_name_lst[isub]
                    if isub==0:
                        phtag='NONE'
                        V_fixed_shift=0
                    else:
                        phtag=utag+' '+str(refpka[isub-1])
                        V_fixed_shift=self.V_bias
                    block_str+=f'      call {iblock} sele {sele_name} end\n'
                    if self.lmds is None:
                       ldin_str+='      ldin {} {:.4f}  0.0  12.0  {:7.2f}  5.0\n'.format(iblock,1/nsubs[isite],self.Gbias.loc[f'lams{isite+1}s{isub+1}','value']+V_fixed_shift)
                    else:
                       ldin_str+='      ldin {} {:19.17f}  0.0  12.0  {:7.2f}  5.0\n'.format(iblock,self.lmds[iblock-2],self.Gbias.loc[f'lams{isite+1}s{isub+1}','value']+V_fixed_shift)
                    msld_str+=f'{isite+1} '
                    iblock+=1
                isite+=1
        msld_str+='fnex 5.5\n'

        excl_lst='      excl '
        iblock=2
        for isite in range(nsites):
            for isub in range(nsubs[isite]):
                shift=1
                for jsub in range(isub+1,nsubs[isite]):
                    excl_lst+='{} {} '.format(iblock,iblock+shift)
                    shift+=1
                iblock+=1
        excl_lst+='\n'

        nldbi=int(5*nblocks*(nblocks-1)/2)
        ldbi_str='      ldbi {}\n'.format(nldbi)
        ibias=0
        iblock0=2
        for isite in range(nsites):
            jblock0=iblock0
            for jsite in range(isite,nsites):
                for isub in range(nsubs[isite]):
                    iblock=iblock0+isub
                    if jsite==isite:
                       jlow=isub+1
                    else:
                       jlow=0
                    for jsub in range(jlow,nsubs[jsite]):
                        jblock=jblock0+jsub
                        ibias+=1
                        ldbi_str+='      ldbv {:3d} {:3d} {:3d}  6   0.0 {:7.2f} 0\n'.format(ibias,iblock,jblock,self.Gbias.loc[f'cs{isite+1}s{isub+1}s{jsite+1}s{jsub+1}','value'])
                        ibias+=1
                        ldbi_str+='      ldbv {:3d} {:3d} {:3d} 10 -5.56 {:7.2f} 0\n'.format(ibias,iblock,jblock,self.Gbias.loc[f'xs{isite+1}s{isub+1}s{jsite+1}s{jsub+1}','value'])
                        ibias+=1
                        ldbi_str+='      ldbv {:3d} {:3d} {:3d} 10 -5.56 {:7.2f} 0\n'.format(ibias,jblock,iblock,self.Gbias.loc[f'xs{jsite+1}s{jsub+1}s{isite+1}s{isub+1}','value'])
                        ibias+=1
                        ldbi_str+='      ldbv {:3d} {:3d} {:3d}  8 0.017 {:7.2f} 0\n'.format(ibias,iblock,jblock,self.Gbias.loc[f'ss{isite+1}s{isub+1}s{jsite+1}s{jsub+1}','value'])
                        ibias+=1
                        ldbi_str+='      ldbv {:3d} {:3d} {:3d}  8 0.017 {:7.2f} 0\n'.format(ibias,jblock,iblock,self.Gbias.loc[f'ss{jsite+1}s{jsub+1}s{isite+1}s{isub+1}','value'])
                jblock0+=nsubs[isite]
            iblock0+=nsubs[isite]

        if soft==False:
           soft_str='\n'
        else:
           if platform=='omm':
              soft_str='somm\n'    # turn on soft-core for VDW in omm
           elif platform=='blade':
              soft_str='soft on\n' # turn on soft-core for VDW and elec in domdec/omm

        block_str+='\n'
        block_str+=excl_lst
        block_str+='\n'
        block_str+='      qldm theta\n'
        block_str+='      lang temp {}\n'.format(tsim)
        block_str+='      pmel ex\n'
        block_str+=soft_str
        block_str+='\n'
        block_str+=ldin_str
        block_str+='\n'
        block_str+='      rmla bond thet impr\n'
        block_str+=msld_str
        block_str+='      msma\n'
        block_str+='\n'
        block_str+=ldbi_str
        block_str+='END\n'
        if debug:
           print(block_str)
        lingo.charmm_script(block_str)

# Ensure that FFT grid is product of small primes (2, 3, 5)
def find_prime_prod(n):
    '''
    find a minimum integer that is a product of 2, 3, or 5, and is greater than the given value n
    '''
    n=np.ceil(n)
    find=0
    while find == 0:
          if (n%2 != 0):
             find=0   # we have to find an even number
          else:
             ni=n
             while ni:
                   #print(n,ni)
                   flag=0
                   for x in (2,3,5):
                       if ni % x == 0:
                          ni = ni / x
                          flag=1
                          # "break" terminates the current for/while loop
                          # and resumes execution at the next statement
                          break
                   if ni==1:
                      find=1
                      return int(n)
                   else:
                      if flag == 1:
                         # "continue" rejects the remaining statement in the current iteration
                         # and returns the control to the beginning of the while loop
                         continue
                      break
          n += 1
    return int(n)

nbond={'elec': True,
       'atom': True,
       'cdie': True,
       'eps': 1,
       'vdw': True,
       'vatom': True,
       'vfswitch': True,
       'cutnb': 15,
       'cutim': 15,
       'ctofnb': 12,
       'ctonnb': 10,
       'ewald': True,
       'pmewald': True,
       'kappa': 0.320,
       'order': 6,
       'fftx': 40,
       'ffty': 40,
       'fftz': 40 
      }

def setup_nb(nb):
    box_dat=np.loadtxt(inp_box_fn,ndmin=1)
    size=box_dat[1]
    fft=find_prime_prod(size)
    nbond['fftx']=fft
    nbond['ffty']=fft
    nbond['fftz']=fft
    nb_pme=pycharmm.NonBondedScript(**nb)
    nb_pme.run()
    if platform=='omm':
       lingo.charmm_script('''
             omm on platform cuda deviceid {}
             '''.format(cuda_device_id))
    elif platform=='blade':
       # from Charlie
       #omp_num_threads=int(os.environ["OMP_NUM_THREADS"])
       #all_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
       #nproc_loc=(len(all_visible_devices)//omp_num_threads)  # // is floor division
       #rank_loc=(rank%nproc_loc)
       #my_visible_devices=all_visible_devices[omp_num_threads*rank_loc:omp_num_threads*(rank+1)]
       ## this is also required. otherwise all BLaDE processes will use the same GPU device 
       #os.environ['CUDA_VISIBLE_DEVICES']=','.join(my_visible_devices)
       os.environ['CUDA_VISIBLE_DEVICES']=str(cuda_device_id)
       lingo.charmm_script('energy')
       lingo.charmm_script('blade on')
    elif platform=='domdec':
       lingo.charmm_script('domdec gpu only dlb off ndir 1 1 1')
    elif platform=='cpu':
        print('ERROR! cpu energy is incorrect b/c we have pme and block')
        quit()
    energy.show()

def setup_pbc():
    box_dat=np.loadtxt(inp_box_fn,ndmin=1)
    size=box_dat[1]
    crystal.define_cubic(size)
    crystal.build(12)
    lingo.charmm_script('coor stat')
    xave=lingo.get_energy_value('XAVE')
    if platform=='omm':
       offset=size*0.5
       # if box center is near the origin, move it to (size*0.5,size*0.5,size*0.5)
       if abs(xave) < size*0.5-xave:
          pos=coor.get_positions()
          coor.set_positions(pos+size*0.5)
    elif platform=='blade' or platform=='cpu':
       offset=0.0
       # if box center is near (size*0.5,size*0.5,size*0.5), move it to the origin
       if xave-offset > abs(xave-size*0.5):
          pos=coor.get_positions()
          coor.set_positions(pos-size*0.5)
    image.setup_segment(offset,offset,offset,segid)
    for i in "TIP3 SOD CLA CAL".strip().split():
        image.setup_residue(offset,offset,offset,i)
    lingo.charmm_script('coor stat')
    xyzcomp=coor.get_positions()
    xyzcomp['w']=xyzcomp.x
    coor.set_comparison(xyzcomp)

def mini():
    if platform=='omm':
       minimize.run_omm(nstep=50,nprint=50,tolenr=1e-3,tolgrd=1e-3)
       lingo.charmm_script('energy omm')
    elif platform=='blade' or platform=='cpu':
       # cannot run mini using blade?
       minimize.run_sd(nstep=50,nprint=50,tolenr=1e-3,tolgrd=1e-3)
    # write coordinates
    write.coor_pdb(out_dir+'/mini.pdb')

class scat_restrain:
    '''
    restrain all or heavy atoms between analogous groups of the 2 or 3 protonation states
    using SCAT restraint.
    a note from documentation: "Scaling of constrained atoms is only tested within domdec."
    '''
    def __init__(self,titr_res_dict,heavy=True):
        self.titr_res_dict=titr_res_dict
        self.heavy=heavy
    def restrain(self):
        scat_str='BLOCK\n'
        scat_str+='      ! enables Scaling of Constrained AToms\n'
        scat_str+='      scat on\n'
        scat_str+='      scat k 300\n'
        self.cats_str=''

        isite=0
        for aa in self.titr_res_dict.keys():
            resn=aa.upper()
            atom_list=titr_grp(resn)
            for ires in self.titr_res_dict[resn][0]:
                resid=str(ires)
                for atom_name in atom_list:
                    if self.heavy==True:  # restrain heavy atoms
                       if atom_name[0]!='H':
                          self.apply_restrain(resid,atom_name)
                    else:                 # restrain all atoms
                       self.apply_restrain(resid,atom_name)

                isite+=1

        #print(self.cats_str)
        scat_str+=self.cats_str
        scat_str+='END\n'
        if debug:
           print(scat_str)
        lingo.charmm_script(scat_str)

    def apply_restrain(self,res_id,name):
        sel_w=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=res_id,atom_type=name+'W')
        sel_m=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=res_id,atom_type=name+'M')
        sel_p=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=res_id,atom_type=name+'P')
        sel_u=pycharmm.SelectAtoms().by_res_and_type(seg_id=segid,res_id=res_id,atom_type=name+'U')
        w=sel_w.get_n_selected()
        m=sel_m.get_n_selected()
        p=sel_p.get_n_selected()
        u=sel_u.get_n_selected()
        if w+m+p+u>=2:  # this is required when blade is used, otherwise we will see errors like "Error running curandGenerateNormal(gen, p, n, 0.0f, 1.0f) in file source/blade/src/rng/rng_gpu.cxx, function rand_normal, line 89, Error string: CURAND_STATUS_LAUNCH_FAILURE" 
           self.cats_str+='      cats sele segid {} .and. resid {} .and. ( type {}W .or. type {}M .or. type {}P .or. type {}U ) end\n'.format(segid, res_id, name, name, name, name)

def dyn_init():
    lingo.charmm_script('''
          faster on
    ''')
    shake.on(param=True,tol=1e-7,bonh=True,fast=True)
    if platform=='omm':
       dyn_dict['omm']='gamma 5'
       dyn_dict['blade']=False
       dyn_dict['iunldm']=-1
       dyn_dict['ntrfrq']=5000
    elif platform=='blade':
       n = psf.get_natom()
       scalar.set_fbetas([5.0] * n)
       dyn_dict['omm']=False
       dyn_dict['blade']=True
       dyn_dict['iunldm']=lmd_unit
       dyn_dict['ntrfrq']=0

# NVT simulation
# nsavv greater than zero is not supported with blade
dyn_dict={
    'leap': True,
    'verlet': False,
    'cpt': False,
    'new': False,
    'langevin': True,
    'start': True,
    'timestep': 0.002,
    'nstep': nsteps_per_cyc,
    'nsavc': 0,
    'nsavv': 0,
    'nsavl': nsteps_per_cyc,  # frequency for saving lambda values in lamda-dynamics
    'nprint': nsteps_per_cyc, # Frequency to write to output
    'iprfrq': nsteps_per_cyc, # Frequency to calculate averages
    'isvfrq': nsteps_per_cyc, # Frequency to save restart file
    'ntrfrq': 5000,
    'inbfrq':-1,
    'ihbfrq':0,
    'ilbfrq':0,
    'imgfrq':-1,
    'iunrea':-1,
    'iunwri':rst_unit,
    'iuncrd':-1,
    'iunldm':-1,
    'firstt': tsim,
    'finalt': tsim,
    'tstruct': tsim,
    'tbath': tsim,
    #'prmc': True,
    #'pref': 1.0,
    #'iprsfrq': 25,
    'iasors': 1, # assign velocities
    'iasvel': 1, # method for assignment of velocities during heating & equil when IASORS is nonzero.
                 # This option also controls the initial assignment of velocities 
    'iscale': 0, # not scale velocities on a restart
    'scale': 1,  # scaling factor for velocity scaling
    'ichecw': 0, # not check temperature
    'echeck': -1 # not check energy
}


def nvt(run):
    rst=charmm_file.CharmmFile(file_name=rst_fn,file_unit=rst_unit,read_only=False,formatted='formatted')
    if platform=='blade':
       lfp=charmm_file.CharmmFile(file_name=lmd_fn,file_unit=lmd_unit,read_only=False,formatted=False)

    if itt>=2:
       if run==0:
          dyn_dict['start']=False
          dyn_dict['restart']=True
          dyn_dict['iunrea']=rpr_unit
          rpr=charmm_file.CharmmFile(file_name=rpr_fn,file_unit=rpr_unit,read_only=True,formatted='formatted')
       else:
          dyn_dict['start']=True
          dyn_dict['restart']=False
          dyn_dict['iunrea']=-1


    npt_prod=pycharmm.DynamicsScript(**dyn_dict)
    npt_prod.run()
    rst.close()

    # last snapshot
    write.coor_pdb(pdb_fn)

def get_neighbor(current_cond_id,run_id):
    if run_id%2==0:
       if current_cond_id%2==0:
          neighbor_cond_id=current_cond_id+1
       else:
          neighbor_cond_id=current_cond_id-1
    else:
       if current_cond_id%2==0:
          neighbor_cond_id=current_cond_id-1
       else:
          neighbor_cond_id=current_cond_id+1
    if neighbor_cond_id<0:
       neighbor_cond_id=0
    if neighbor_cond_id>=nproc:
       neighbor_cond_id=nproc-1
    return neighbor_cond_id

def get_neighbor_2d(current_cond_id,run_id):
    """
    exchange in lambda_chrg space for two cycles, and then in pH space for two cycles
    """
    cond_id_ph=int(current_cond_id/nchrg)
    cond_id_temp=current_cond_id%nchrg
    if run_id%4==0:
       if cond_id_temp%2==0:
          neighbor_cond_id=current_cond_id+1
       elif cond_id_temp%2==1:
          neighbor_cond_id=current_cond_id-1
       if neighbor_cond_id<cond_id_ph*nchrg or neighbor_cond_id>=(1+cond_id_ph)*nchrg:
          neighbor_cond_id=current_cond_id
    elif run_id%4==1:
       if cond_id_temp%2==0:
          neighbor_cond_id=current_cond_id-1
       elif cond_id_temp%2==1:
          neighbor_cond_id=current_cond_id+1
       if neighbor_cond_id<cond_id_ph*nchrg or neighbor_cond_id>=(1+cond_id_ph)*nchrg:
          neighbor_cond_id=current_cond_id
    elif run_id%4==2:
       if cond_id_ph%2==0:
          neighbor_cond_id=current_cond_id+nchrg
       elif cond_id_ph%2==1:
          neighbor_cond_id=current_cond_id-nchrg
       if neighbor_cond_id<0 or neighbor_cond_id>=nproc:
          neighbor_cond_id=current_cond_id
    elif run_id%4==3:
       if cond_id_ph%2==0:
          neighbor_cond_id=current_cond_id-nchrg
       elif cond_id_ph%2==1:
          neighbor_cond_id=current_cond_id+nchrg
       if neighbor_cond_id<0 or neighbor_cond_id>=nproc:
          neighbor_cond_id=current_cond_id
    return neighbor_cond_id

class rep_ex:
      """
      perform 2D replica exchange of partial charges and Vbias
      """
      def __init__(self,ncycle='',condid='',lmd='',Vbias='',conditions='',charge_df='',atom_index=''):
          """
          Parameters
          ----------
          ncycle : int
              number of exchange cycles
          condid : int
              condition ID
          lmd : float
              charge scaling factor
          Vbias : float
              shift of fixed term of V_bias
          conditions : numpy.array
              a 2D array of with (V_bias, lambda_chrg) for each condition,
          charge_df : pandas.DataFrame
              original charges of all atoms
          atom_index : np.array
              atom index of the flexible region
          """
          self.cond_id=int(condid)
          self.cond_array=conditions
          self.lmd=float(lmd)
          self.Vbias=float(Vbias)
          self.nrun=int(ncycle)
          self.chrg_df=charge_df
          self.inx_flex=atom_index
      def run(self):
          self.open_files()
          for i in range(0,self.nrun):
              #print("run {}, rank {} lambda is {}".format(i,rank,self.lmd))
              nvt(i)
              self.write_trj(i)
              self.cond_data=self.gather_condid(i)
              self.Vbias,self.lmd=self.swap_neighbor_2d(i)
              if debug:
                 print("after exchange, rank {} lambda is {} Vbias is {}".format(rank,self.lmd,self.Vbias))
              self.get_ener_new()

              self.metropolis()
              if debug:
                 print('run ',i, 'cond ',self.cond_id,'Accept',self.Accept)
              self.write_exch_all(i)

              self.update_cond_2d()
              if debug:
                 print("after metropolis, rank {} lambda is {} Vbias is {}".format(rank,self.lmd,self.Vbias))

          self.close_files()
      def open_files(self):
          # all exchange information
          if rank==0:
             self.exch_all=open(exch_all_fn,'w',buffering=1)
             self.exch_all.write('# replica temp. ener. neighbor ntemp nene prob p success? newrep\n')
          # exchange information for each replica
          self.exch=open(exc_fn,'w',buffering=1)
          self.exch.write('#run repl_id ener_im neighbor_repl_id ener_in prob rand accept replica_next\n')
          # condition history for each replica
          self.hist=open(his_fn,'w',buffering=1)
          # dcd for each replica
          self.dcd=charmm_file.CharmmFile(file_name=dcd_fn,file_unit=dcd_unit,read_only=False,formatted=False)
          lingo.charmm_script('''
          traj IWRITE {} NWRITE 1 NFILE {}
          * title
          *
          '''.format(dcd_unit,self.nrun))
          # lambda values for each replica 
          self.lmd_fp=open(lmd_dat_fn,'w',buffering=1)
      def close_files(self):
          if rank==0:
             self.exch_all.close()
          self.exch.close()
          self.hist.close()
          self.dcd.close()
          self.lmd_fp.close()
          comm.barrier()
      def write_trj(self,run):
          lingo.charmm_script('traj write')
          # In theory I need to explicitly call "ENER", b/c we used shake in the dynamics.
          # H-containing bonds contribute "ENER", but not "DYNA".
          # However, even if we don't call "ENER", there is no problem in practice.
          # 1) energies from 'DYNA' and 'ENER' are highly similar for a snapshot
          # in a dynamics run (except for the initial PDB structure).
          # 2) the bond energies won't contribute to metropolis criterion.
          # energy.show()
          self.ener_im=lingo.get_energy_value('ENER')
          self.exch.write('%5d %2d %15f '%(run,repl_id,self.ener_im))
          self.hist.write('%i %s %s \n'%(run,self.Vbias,self.lmd))
          # save lambda values of the last snapshot
          self.lmd_array=self.get_lambda()
          comm.barrier()
      def get_lambda(self):
          fp=FortranFile(lmd_fn,'r')

          # The header and icntrl array are read in as a single record
          # Read the icntrl array (length 20) and extract key variables
          header = (fp.read_record([('hdr',np.string_,4),('icntrl',np.int32,20)]))
          hdr = header['hdr'][0]
          icntrl = header['icntrl'][0][:]
          nfile = icntrl[0]     # Total number of dynamcis steps in lambda file
          npriv = icntrl[1]     # Number of steps preceding this run
          nsavl = icntrl[2]     # Save frequency for lambda in file
          nblk = icntrl[6]      # Total number of blocks = env + subsite blocks
          nsitemld = icntrl[10] # Total number of substitution sites (R-groups) in MSLD

          # Time step for dynamics in AKMA units
          delta4 = (fp.read_record(dtype=np.float32))
          
          # Title in trajectoory file 
          title = (fp.read_record([('h',np.int32,1),('title',np.string_,80)]))[0][1]
          
          # Unused in current processing
          nbiasv = (fp.read_record(dtype=np.int32))
          junk = (fp.read_record(dtype=np.float32))
          
          # Array (length nblk) indicating which subsites below
          # to which R-substitiution site
          isitemld = (fp.read_record(dtype=np.int32))
          
          # Temeprature used in lambda dynamics thermostat
          temp = (fp.read_record(dtype=np.float32))
          
          # Unsed data for this processing
          junk3 = (fp.read_record(dtype=np.float32))
          
          Lambda=np.zeros((nfile,nblk-1))

          for i in range(nfile):
            # Read a line of lambda values
            lambdav = (fp.read_record(dtype=np.float32))
            theta = (fp.read_record(dtype=np.float32))
            Lambda[i,:]=lambdav[1:]
            #print(i,lambdav,theta,Lambda[i,:])
          fp.close()

          if debug:
             print(Lambda)
          # write lambdas
          for i in Lambda[0]:
              self.lmd_fp.write('%.7f '%(i))
          self.lmd_fp.write('\n')

          return Lambda[0]

      def gather_condid(self,run):
          data=self.cond_id   
          data=comm.gather(data,root=0)
          data=comm.bcast(data,root=0)
          if debug:
             print('run',run,'rank',rank,data)
          return data
      def swap_neighbor_2d(self,run):
          self.new_cond_id=get_neighbor_2d(self.cond_id,run)
          self.neighbor_repl_id=self.cond_data.index(self.new_cond_id)
          ph=self.cond_array[self.new_cond_id][0]
          lmd=self.cond_array[self.new_cond_id][1]
          return ph,lmd
      def get_ener_new(self):
          '''
          Reset partial charges and update Vbias value to get new potential energy.
          '''
          chrg_df_new_t=self.chrg_df.copy(deep=True)
          chrg_df_new_t[self.inx_flex]=chrg_df_new_t[self.inx_flex]*self.lmd
          chrg_df_new=chrg_df_new_t.round(ndec)
          psf.set_charge(list(chrg_df_new['chrg']))
          if debug:
             energy.show()

          blk=setup_block(var_df,titr_res,self.Vbias,self.lmd_array)
          blk.update_vbias(self.lmd_array)

          energy.show()
          self.ener_in=lingo.get_energy_value('ENER')
          self.exch.write('%2d %15f '%(self.neighbor_repl_id,self.ener_in))
          comm.barrier()
      def metropolis(self):
          self.Accept=True
          self.prob=1
          self.rand=0
          if self.cond_id > self.new_cond_id:
             comm.send(self.ener_im, dest=self.neighbor_repl_id, tag=11)
             comm.send(self.ener_in, dest=self.neighbor_repl_id, tag=12)
          elif self.cond_id < self.new_cond_id:
             ener_jn=comm.recv(source=self.neighbor_repl_id,tag=11)
             ener_jm=comm.recv(source=self.neighbor_repl_id,tag=12)
             delta=-1*beta*(ener_jm+self.ener_in-self.ener_im-ener_jn)
             #print("beta:",beta,"delta:",delta,"ener_jm:",ener_jm,"ener_in:",self.ener_in,"ener_im:",self.ener_im,"ener_jn:",ener_jn)
             if delta<0:
                self.prob=np.exp(delta)
                #print('prob ',self.prob)
                self.rand=np.random.uniform(low=0.0, high=1.0)
                if self.prob < self.rand:
                   self.Accept=False
                   #print('reject')
          comm.barrier()
          if self.cond_id < self.new_cond_id:
             comm.send(self.Accept,dest=self.neighbor_repl_id,tag=13)
          elif self.cond_id > self.new_cond_id:
             self.Accept=comm.recv(source=self.neighbor_repl_id,tag=13)
          comm.barrier()
          self.exch.write('%5f %5f %5s '%(self.prob,self.rand,str(self.Accept)[0]))
      def write_exch_all(self,run):
          # exchange file for each replica
          if self.Accept == True:
             replica_next=self.neighbor_repl_id
          else:
             replica_next=repl_id
          self.exch.write('%2d \n'%(replica_next))
          exch_str='{:2d} {:12.6f} {:15.6f} {:2d} {:12.6f} {:15.6f} {:5.3f} {:5.3f} {:1s} {:2d}\n'.format(repl_id,self.cond_id,self.ener_im,self.neighbor_repl_id,self.new_cond_id,self.ener_in,self.prob,self.rand,str(self.Accept)[0],replica_next)
          # gather all exchange info (a string) to an array
          comm.barrier()
          exch_str_all=comm.gather(exch_str,root=0)
          comm.barrier()
          if rank==0:
             exch_list_all=[]
             for i in range(0,nproc):
                 exch_list_all.append(exch_str_all[i].split())
             exch_all_array=np.array(exch_list_all)
             exch_array_sort=pd.DataFrame(exch_all_array,columns=['repl_id','condid_m','ener_im','neighbor_repl_id','condid_n','ener_in','prob','rand','accept','neighbor_repl_id']).astype({'condid_m':float}).sort_values(by='condid_m',ascending=True).to_numpy()
             self.exch_all.write('# Exchange %15d: STEP %12d: REPEAT     1\n'%(run+1,run+1))
             for i in range(0,nproc):
                 self.exch_all.write('%2d %12f %15f %2d %12f %15f %5.3f %5.3f %1s %2d\n'%
                 (int(exch_array_sort[i,0])+1,float(exch_array_sort[i,1]),float(exch_array_sort[i,2]),
                  int(exch_array_sort[i,3])+1,float(exch_array_sort[i,4]),float(exch_array_sort[i,5]),
                  float(exch_array_sort[i,6]),float(exch_array_sort[i,7]),exch_array_sort[i,8],int(exch_array_sort[i,9])+1))
      def update_cond_2d(self):
          # update condition 
          if self.Accept == False:
             self.new_cond_id=self.cond_id
             self.Vbias=self.cond_array[self.new_cond_id][0]
             self.lmd=self.cond_array[self.new_cond_id][1]
             # reset partial charges to old values
             chrg_df_new_t=self.chrg_df.copy(deep=True)
             chrg_df_new_t[self.inx_flex]=chrg_df_new_t[self.inx_flex]*self.lmd
             chrg_df_new=chrg_df_new_t.round(ndec)
             psf.set_charge(list(chrg_df_new['chrg']))

             blk=setup_block(var_df,titr_res,self.Vbias,self.lmd_array)
             blk.update_vbias(self.lmd_array)

             energy.show()
          else:
             self.lmd=self.lmd
             self.Vbias=self.Vbias
          self.cond_id=self.new_cond_id
      def trj_unmixing(self):
          if rank==0:
             #lingo.charmm_script('prnlev 10')
             exch_all=charmm_file.CharmmFile(file_name=exch_all_fn,read_only=True,formatted=True)
             for i in range(0,nproc):
                 unit_inp=7+i
                 unit_out=48+i
                 fn_inp='aa'+str(i)+'/dcd/'+name+'_prod'+str(itt)+'.dcd'
                 os.system('mkdir -p cond'+str(i)+'/dcd')
                 fn_out='cond'+str(i)+'/dcd/'+name+'_prod'+str(itt)+'.dcd'
                 lingo.charmm_script('open read file unit {} name {}'.format(unit_inp,fn_inp))
                 lingo.charmm_script('open writ file unit {} name {}'.format(unit_out,fn_out))
             lingo.charmm_script('merge firstu 7 nunit {} outp 48 RTOTemp excu {} NEXChange {} nrpl {} nrep 1'.format(nproc,exch_all.file_unit,ncycles,nproc))
             exch_all.close
      def accept_ratio(self):
          if rank==0:
             dat=np.loadtxt(exch_all_fn,usecols=(1,4,8),dtype={'names':('i','j','accept'),'formats':(float,float,'|S1')})
             df=pd.DataFrame(dat)
             #print(df)
             df_ex=df.groupby(['i','j']).size().reset_index(name='allex')
             df_ac=df.groupby(['i','j','accept']).size().reset_index(name='success')
             df_ac_t=df_ac[df_ac['accept']==b'T'].reset_index()
             print(df_ex)
             print(df_ac_t)
             npairs=df_ex.shape[0]
             ratio_list=[]
             #print(npairs)
             for i in range(0,npairs):
                 cond_i=df_ex.loc[i,['i']][0]
                 cond_j=df_ex.loc[i,['j']][0]
                 if cond_i < cond_j:
                    if np.sum((df_ac_t['i']==cond_i) & (df_ac_t['j']==cond_j))==0:
                       #print(cond_i,cond_j,'no successful exchange')
                       accepted=0
                    else:
                       accepted=df_ac_t[(df_ac_t['i']==cond_i) & (df_ac_t['j']==cond_j)]['success'].to_numpy()[0]
                    exchanged=df_ex[(df_ex['i']==cond_i) & (df_ex['j']==cond_j)]['allex'].to_numpy()[0]
                    ratio=accepted/exchanged
                    #print(cond_i,cond_j,accepted,exchanged,ratio)
                    ratio_list.append([cond_i,cond_j,accepted,exchanged,ratio])
             ratio_array=np.array(ratio_list)
             #print(ratio_array)
             np.savetxt(ratio_fn,ratio_array,fmt='%.6f')

def get_latest_lmd(cond_arr):
    """
    before restarting a replica exchange simulation, get the latest state

    Parameters
    ----------
    cond_arr : numpy.array
        a 2D array of with (pH, lambda_chrg) for each condition,

    Returns
    -------
    pH : float
        pH of a replica
    l : float
        charge scaling factor of a replica
    condid: int
        condition ID of a replica
    """
    exch_all_pre=np.loadtxt(exch_all_pre_fn,usecols=(9),dtype=int)
    repid_new=exch_all_pre[-1*nproc:]-1
    pH=cond_arr[repid_new==rank][0][0]
    l=cond_arr[repid_new==rank][0][1]
    condid=np.where(repid_new==rank)[0][0]
    if debug:
       print(rank+1,l,pH,condid)
    return pH,l,condid

def main():
    global repl_id, exch_all_pre_fn
    exch_all_pre_fn=wrk_dir+'/exchange-all-'+str(ittm1)+'.dat'
    cond_arr=lambda_schedule(n_chrg=nchrg,delta_fixed=1.5,n_bias=nbias,linear_chrg=True,geometric_chrg=False,sin_chrg=False)
    if debug:
       print(cond_arr)
    work=distributeWork(cond_arr.shape[0])
    if debug:
       print(work)
    for i in work[rank]:
        if len(work[rank])>1:
           print('''Error! 
In the current implementation of replica exchange,
one process cannot handle more than two replicas.
Please allocate enough resources.''')
        if itt >= 2:
           Vbias,l,condid=get_latest_lmd(cond_arr)
           repl_id=rank
        else:
           condid=i
           Vbias=cond_arr[i][0]
           l=cond_arr[i][1]
           repl_id=condid
        if debug:
            print(f'rank: {rank}, replica_id: {repl_id}, initial cond_id: {condid}, lambda: {l:.4f}, Vbias: {Vbias:.4f}')
        global out_dir
        out_dir='aa'+str(repl_id)
        setup_dir()
        ################################################################
        # specify file names
        global dcd_fn, rst_fn, pdb_fn, crd_fn, rpr_fn, log_fn
        global exc_fn, his_fn, exch_all_fn, ratio_fn
        global lmd_fn, lmd_dat_fn
        dcd_fn=out_dir+'/dcd/'+name+'_prod'+str(itt)+'.dcd'
        rst_fn=out_dir+'/res/'+name+'_prod'+str(itt)+'.res'
        rpr_fn=out_dir+'/res/'+name+'_prod'+str(ittm1)+'.res'
        lmd_fn=out_dir+'/res/'+name+'_prod'+str(itt)+'.lmd'
        lmd_dat_fn=out_dir+'/res/'+name+'_prod'+str(itt)+'.lmd.dat'
        pdb_fn=out_dir+'/pdb/prod'+str(itt)+'.pdb'
        log_fn=out_dir+'/out/prod'+str(itt)+'.out'
        exc_fn=out_dir+'/exc/exch'+str(itt)+'.dat'
        his_fn=out_dir+'/his/cond_his'+str(itt)+'.dat'
        exch_all_fn=out_dir+'/../exchange-all-'+str(itt)+'.dat'
        ratio_fn=out_dir+'/../ratio-'+str(itt)+'.dat'
        ################################################################
        charmm_log=open_clog(log_fn)
        read_param()

        global var_df,titr_res,nsubs,nblocks,nsites

        with open(titr_res_fn) as f:
             data=f.read()
        titr_res=json.loads(data)

        nsubs=[]
        # https://stackoverflow.com/questions/5629023/order-of-keys-in-dictionaries-in-old-versions-of-python
        # In Python 3.7.0 the insertion-order preservation nature of dict objects 
        # has been declared to be an official part of the Python language spec.
        for aa in titr_res.keys():
            resn=aa.upper()
            nsub=2
            if int(ns)==3:
               if resn == 'ASP' or resn == 'GLU':
                  nsub=3
            for ires in titr_res[resn][0]:
                nsubs.append(nsub)
        nsubs=np.array(nsubs)
        nblocks=np.sum(nsubs)
        nsites=len(nsubs)

        # save these file for ALF
        if int(itt)==1:
           np.savetxt(wrk_dir+'/nsubs',nsubs[None],fmt='%d')
           np.savetxt(wrk_dir+'/nblocks',np.array([nblocks]),fmt='%d')
           np.savetxt(wrk_dir+'/name',np.array([comp]),fmt='%s')
           np.savetxt(wrk_dir+'/ncentral',np.array([0]),fmt='%d')
           np.savetxt(wrk_dir+'/nnodes',np.array([1]),fmt='%d')
           np.savetxt(wrk_dir+'/nreps',np.array([1]),fmt='%d')
           np.savetxt(wrk_dir+'/ntersiteflat',np.array([0,1])[None],fmt='%d')
           np.savetxt(wrk_dir+'/ntersiteprod',np.array([0,1])[None],fmt='%d')


        chrg_orig,index_flex=read_init()

        # BLaDE requires PBC
        setup_pbc()
        var_df=read_var(var_fn)
        blk=setup_block(var_df,titr_res,Vbias)
        blk.run()

        setup_nb(nbond)

        # When we assign a DataFrame to a new variable using =, 
        # we are NOT creating a new copy of the DataFrame. 
        # We are merely adding a new name to call the same object. 

        # scale partial charges
        chrg_df_new_t=chrg_orig.copy(deep=True)
        # only keep 4 digits after the decimal point
        chrg_df_new_t[index_flex]=chrg_df_new_t[index_flex]*l
        chrg_df_new=chrg_df_new_t.round(ndec)
        psf.set_charge(list(chrg_df_new['chrg']))
        energy.show()

        myscat=scat_restrain(titr_res,heavy=True)
        myscat.restrain()

        dyn_init()
        rex=rep_ex(ncycle=ncycles,condid=condid,lmd=l,Vbias=Vbias,conditions=cond_arr,charge_df=chrg_orig,atom_index=index_flex)
        rex.run()
        close_clog(charmm_log)

        #rex.trj_unmixing()
        rex.accept_ratio()

    comm.barrier()
    MPI.Finalize()
    exit()

if __name__ == '__main__':
   main()
