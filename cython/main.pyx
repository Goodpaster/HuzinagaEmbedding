"""
Huzinaga Freeze-and-Thaw Algorithm

This module interfaces with pyscf to implement the Huzinaga level-shift
projection operator embedding method.

By Dhabih Chulahi and Daniel Graham
"""

from __future__ import print_function, division
import re
from read_input import read_input
from pyscf import cc, gto, scf
from pyscf.tools import cubegen
from copy import deepcopy as copy
from functools import reduce
import simple_timer
import h5py
import gzip
import os
from integrals import get_interaction_energy, concatenate_mols
from integrals import gen_grids, get_delta_den
from scf import get_scf
from scf import do_embedding
from scf import do_supermol_scf
from scf import do_final_dft_embed
from purify import McWeeny_pur, Truf_pur, Palser_pur, Nat_orb_mat
from localize import get_sub2sup
from cpython cimport bool

import numpy as np
cimport numpy as np
import scipy as sp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython
@cython.boundscheck(False)
def main(filename):
    """The main algorithm interface to determine properties based on Huzinaga
    operator.

    Parameters
    ----------
    filename : str
        The name of the input file

    Returns
    -------
    inp : read_input object
        The final read_input object with final attributes.

    """
    # Print header and input file.
    pstr ("", delim="*", addline=False)
    pstr (" FREEZE-AND-THAW CALCULATION ", delim="*", fill=False, addline=False)
    pstr ("", delim="*", addline=False)
    pstr ("Input File")
    [print (i[:-1]) for i in open(filename).readlines() if ((i[0] not in ['#', '!'])
        and (i[0:2] not in ['::', '//']))]
    pstr ("End Input", addline=False)

    # Read input file and start timing.
    inp = read_input(filename)
    inp.timer = simple_timer.timer()
    cdef int nsub = inp.nsubsys
    inp.DIIS = None
    inp.ndiags = [0 for i in range(nsub)]

    # Generate supermolecular grid for use in use in subsystem grids.
    inp.grids = gen_grids(inp)
    cdef int sna = inp.smol.nao_nr()
    na = [inp.mol[i].nao_nr() for i in range(nsub)]

    # Initialize 1e matrices from supermolecular system.
    inp.timer.start("OVERLAP INTEGRALS")
    pstr ("Getting subsystem overlap integrals", delim=" ")
    inp.sSCF = get_scf(inp, inp.smol)
    cdef np.ndarray[DTYPE_t, ndim=2] Hcore = inp.sSCF.get_hcore()
    cdef np.ndarray[DTYPE_t, ndim=2] Smat = inp.sSCF.get_ovlp()
    cdef np.ndarray[DTYPE_t, ndim=2] Fock = np.zeros((sna, sna))
    Dmat = [None for i in range(nsub)]
    inp.timer.end("OVERLAP INTEGRALS")

    # Initialize subsystem densities.
    inp.timer.start("INITIAL SUBSYSTEMS")
    inp.mSCF = [None for i in range(nsub)]
    inp.Fermi = np.zeros((inp.nsubsys))
    pstr ('Initial subsystem densities', delim=' ')
    for i in range(nsub):
        inp.mSCF[i] = get_scf(inp, inp.mol[i])

    
    # Check if checkpoint file exists and read density and Fermi energies.
    if inp.readchk and os.path.isfile(inp.filename.split('.')[0]+'.hdf5'):
        print ("Reading initial densities from file: {0}".format(inp.filename.split(',')[0]+'.hdf5'))
        inp.h5pyname = inp.filename.split('.')[0] + ".hdf5"
        with h5py.File(inp.h5pyname, 'r') as hf:
            for j in range(nsub):
                dmatname = 'Dmat[{0}]'.format(j)
                if dmatname in hf:
                    Dmat[j] = hf['Dmat[{0}]'.format(j)][:]
                else:
                    Dmat[j] = inp.mSCF[j].init_guess_by_atom()
                    #Dmat[i] = inp.mSCF[i].init_guess_by_minao()
                    hf.create_dataset('Dmat[{0}]'.format(j), data=Dmat[j])

            if 'Fermi' in hf:
                inp.Fermi = hf['Fermi'][:]
            else:
                hf.create_dataset('Fermi', data=inp.Fermi)

    # Generate initial subsystem guess if not reading checkpoint file.
    else:
        if inp.readchk:
            print ("Checkpoint .hdf5 not found")
        # Initialize HDFS file.
        if inp.filename[-4:] == '.inp':
            inp.h5pyname = inp.filename[:-4]+'.hdf5'
        else:
            inp.h5pyname = inp.filename+'.hdf5'

        print ("Guessing initial subsystem densities")
        # Save density matrix to file for later read/write
        with h5py.File(inp.h5pyname, 'w') as hf:
            for j in range(nsub):
                Dmat[j] = inp.mSCF[j].init_guess_by_atom()
                #Dmat[i] = inp.mSCF[i].init_guess_by_minao()
                hf.create_dataset('Dmat[{0}]'.format(j), data=Dmat[j])
            hf.create_dataset('Fermi', data=inp.Fermi)
  
    # WHAT IS THIS? 
    inp.sub2sup = get_sub2sup(inp, inp.mSCF)
    inp.timer.end("INITIAL SUBSYSTEMS")

    # Calculate supermolecular energy and localize orbitals.
    if inp.embed.localize:
        pstr ("Supermolecular DFT")
        inp.sSCF = do_supermol_scf(inp, Dmat, Smat)
        from localize import localize
        pstr ("Localizing Orbitals", delim="=")
        Dmat = localize (inp, inp.sSCF, inp.mSCF, Dmat, Smat)

    # Initialize WF-in-DFT cycles
    Ediff = None
    if inp.wfembed:
        wf_max_cycles = inp.wfembed.cycles
        wf_conv_criteria = inp.wfembed.conv
        pstr("Starting WFT-in-DFT Cycles")
    else:
        wf_max_cycles = 1
        wf_conv_criteria = 1.

    wf_cycle = 0
    wf_err = 10.
    while(wf_cycle < wf_max_cycles and wf_err > wf_conv_criteria):

        old_Dmat = np.copy(Dmat)

        # Initialize DFT-in-DFT freeze and thaw cycles.
        inp.timer.start("FREEZE AND THAW")
        if inp.embed.cycles == 1:
            smethod = "SCF"
        else:
            smethod = "Freeze-and-Thaw"
        if inp.embed.cycles > 0:
            pstr ("Starting "+smethod)
        inp.error = 1.
        inp.ift = 0
        while ((inp.error > inp.embed.conv) and (inp.ift < inp.embed.cycles)):

            inp.ift += 1
            inp.prev_error = copy(inp.error)
            inp.error = 0.

            # Cycle active subsystems
            for a in range(nsub):
                if inp.embed.freezeb and a > 0: continue
                # Freeze WF subsystem density during WF-in-DFT cycles.
                if inp.wfembed and wf_cycle > 0 and a == 0: continue

                # Embed active subsystem.
                inp.timer.start("TOT. EMBED. OF {0}".format(a+1))
                Fock, inp.mSCF[a], Dnew, eproj, err = do_embedding(a, inp,
                    inp.mSCF, Hcore, Dmat, Smat, Fock, cycles=inp.embed.subcycles,
                    conv=inp.embed.conv, llast=False)
                er = sp.linalg.norm(Dnew - Dmat[a])
                Dmat[a] = np.copy(Dnew)
                print (' iter: {0:>3d}:{1:<2d}       |ddm|: {2:12.6e}     '
                       'Tr[DP] {3:13.6e}'.format(inp.ift, a+1, er, eproj))
                inp.error += er
                inp.timer.end("TOT. EMBED. OF {0}".format(a+1))


        # Update density and Fermi enery levels in h5 checkpoint file.
        with h5py.File(inp.h5pyname, 'r+') as hf:
            for j in range(nsub):
                dData = hf['Dmat[{0}]'.format(j)]
                dData[...] = Dmat[j]
            fData = hf['Fermi']
            fData[...] = inp.Fermi


        # Determine DFT-in-DFT freeze and thaw cycle convergence.
        if ((inp.embed.cycles > 1 and inp.error < inp.embed.conv)
           or (inp.embed.cycles == 1 and err < inp.embed.conv)):
            pstr (smethod+" converged!", delim=" ", addline=False)
        elif inp.embed.cycles == 0:
            pass
        else:
            pstr (smethod+" NOT converged!", delim=" ", addline=False)
    
        # Embed final cycle at high level of theory.
        if inp.embed.cycles > -1:
            inp.timer.start("TOT. EMBED. OF 1")
            maxiter = inp.embed.subcycles
            conv = inp.embed.conv
            if inp.embed.method != inp.method:
                pstr ("Subsystem 1 at higher mean-field method", delim="=")
                maxiter = inp.maxiter
                conv = inp.conv
    
            Fock, mSCF_A, DA, eproj, err = do_embedding(0, inp, inp.mSCF, Hcore,
                Dmat, Smat, Fock, cycles=maxiter, conv=conv, llast=True)
    
            er = sp.linalg.norm(DA - Dmat[0])
            print (' iter: {0:>3d}:{1:<2d}       |ddm|: {2:12.6e}     '
                   'Tr[DP] {3:13.6e}'.format(inp.ift+1, 1, er, eproj))
            inp.timer.end("TOT. EMBED. OF 1")
    
            if inp.embed.method != inp.method:
                inp.eHI = mSCF_A.e_tot
            else:
                inp.eHI = None
        inp.timer.end("FREEZE AND THAW")
    
        # Calculate WF properties
        if inp.method == 'ccsd' or inp.method == 'ccsd(t)':
            inp.timer.start("CCSD")
            pstr ("Doing CCSD Calculation on Subsystem 1")
            mCCSD = cc.CCSD(mSCF_A)
            ecc, t1, t2 = mCCSD.kernel()
            inp.eHI = mSCF_A.e_tot + ecc
    
            # Get MO WFT density for embedding WF-in-DFT cycles
            if inp.wfembed:
                Dnew = mCCSD.make_rdm1()
            inp.timer.end("CCSD")
            print ("Hartree-Fock Energy:  {0:19.14f}".format(mSCF_A.e_tot))
            print ("CCSD Correlation:     {0:19.14f}".format(ecc))
            print ("CCSD Energy:          {0:19.14f}".format(mSCF_A.e_tot+ecc))
    
        if inp.method == 'ccsd(t)':
            from pyscf.cc import ccsd_t, ccsd_t_lambda_slow, ccsd_t_rdm_slow
            inp.timer.start("CCSD(T)")
            pstr ("Doing CCSD(T) Calculation on Subsystem 1")
            ecc += ccsd_t.kernel(mCCSD, mCCSD.ao2mo())
            inp.eHI = mSCF_A.e_tot + ecc

            # Get MO WFT density for embedding in WF-in-DFT cycles
            if inp.wfembed:
                eris = mCCSD.ao2mo()
                l1, l2 = ccsd_t_lambda_slow.kernel(mCCSD, eris, t1, t2)[1:]
                Dnew = ccsd_t_rdm_slow.make_rdm1(mCCSD, t1, t2, l1, l2, eris=eris)
            inp.timer.end("CCSD(T)")
            print ("CCSD(T) Correlation:     {0:19.14f}".format(ecc))
            print ("CCSD(T) Energy:          {0:19.14f}".format(mSCF_A.e_tot+ecc))
    
        if re.match(re.compile('cas\[.*\].*'), inp.method):
            from pyscf import mcscf, mrpt
            space = [int(i) for i in (inp.method[inp.method.find("[") + 1:inp.method.find("]")]).split(',')]
            inp.timer.start('CAS')
            moc = mSCF_A.mo_coeff
            mcSCF = mcscf.CASSCF(mSCF_A, space[0], space[1])
            mcSCF.max_cycle_macro = 150
            ecas = mcSCF.kernel(moc)[0]
            inp.timer.end('CAS')
            inp.eHI = ecas
            print ("Hartree-Fock Energy:  {0:19.14f}".format(mSCF_A.e_tot))
            print ("CAS Correlation:     {0:19.14f}".format(ecas - mSCF_A.e_tot))
            print ("CAS Energy:          {0:19.14f}".format(ecas))
            if ("nevpt2" in inp.method):
                inp.timer.start('NEVPT2')
                nev = mrpt.nevpt2.NEVPT(mcSCF)
                eNEV = nev.kernel()
                print ("NEVPT2 = {0:20.15f}".format(eNEV))
                print ("Total NEVPT2 = {0:20.15f}".format(ecas + eNEV))
                inp.timer.end('NEVPT2')
                inp.eHI = ecas + eNEV

                # Currently untested density.
                if inp.wfembed:
                    print ("NEVPT2 Density untested")
                    Dnew = nev.onerdm

            # Generate MO WF density for embedding WF-in-DFT cycles 
            # or orbital visualization
            if inp.wfembed or inp.gencube == "cas":
                casdm1 = mcscf.make_rdm1(mcSCF)
                Dnew = casdm1
                #nmo = mcSCF.mo_coeff.shape[1]
                #nocc = mcSCF.ncas + mcSCF.ncore
                #casdm1_mo = np.zeros((nmo, nmo))
                #idx = np.arange(mcSCF.ncore) 
                #casdm1_mo[idx, idx] = 2
                #casdm1_mo[mcSCF.ncore:nocc, mcSCF.ncore:nocc] = casdm1
                ##casdm1 = mcscf.make_rdm1(mcSCF)
                ## mocas = mcSCF.mo_coeff[:,mcSCF.ncore:mcSCF.ncore + mcSCF.ncas]
                ## casdmao = reduce(np.dot, (mocas, casdm1, mocas.T))
                #Dnew = casdm1_mo

            # Generate density cube files for orbital visualization
            if inp.gencube == "cas":
                inp.timer.start("Generate Cube file")
                molCAS_cubename = filename.split(".")[0] + "_4CAS.cube"
                molCASfull_cubename = filename.split(".")[0] + "_fullCAS.cube"
                cubegen.density(inp.mol[0], molCASfull_cubename, casdmao)
    
                For inner orbitals 
                ncore = mcSCF.ncore
                ncas = mcSCF.ncas
                nocc = ncore + ncas
                mo_occ =  np.zeros_like(mSCF_A.mo_occ)
                for i in range(ncas):
                    mo_occ[ncore + i] = 2.0
                    den = scf.hf.make_rdm1(mcSCF.mo_coeff, mo_occ)
                    mo_occ.fill(0)
                cubegen.density(inp.mol[0], molCAS_cubename, mcSCF
                inp.timer.end("Generate Cube file")
    
        if inp.method == 'fci':
            from pyscf import fci
            inp.timer.start("FCI")
            pstr ("Doing FCI Calculation on Subsystem 1")
            cisolver = fci.FCI(inp.mol[0], mSCF_A.mo_coeff)
            efci = cisolver.kernel()[0]
            inp.eHI = efci

            # Not available.
            if inp.wfembed:
                print ("FCI Density not available")
                Dnew = old_Dmat[0]
            inp.timer.end("FCI")
            print ("Hartree-Fock Energy:  {0:19.14f}".format(mSCF_A.e_tot))
            print ("FCI Correlation:     {0:19.14f}".format(efci - mSCF_A.e_tot))
            print ("FCI Energy:          {0:19.14f}".format(efci))

        # Prepare density matrix for WF-in-DFT cycles.
        if inp.wfembed:
            if wf_cycle < 1:
                # Get interaction energy between subsystems.
                # Get supermolecular energy.
                inp.timer.start("INTERACTION ENERGIES")
                ldosup = not inp.embed.localize
                if ldosup and wf_cycle < 1: pstr ("Supermolecular DFT")
                Etot, EA = get_interaction_energy(inp, Dmat, Smat, supMol=True, ldosup=ldosup)
                if not Ediff:
                    Ediff = inp.Esup - Etot
                inp.timer.end("INTERACTION ENERGIES")

            pstr ("Density Matrix Purification")
            inp.timer.start("DMP")
            if re.match(re.compile('cas\[.*\].*'), inp.method):
                Dmat[0] = np.copy(Dnew)
            elif inp.wfembed.natorb:
                Dmat[0] = Nat_orb_mat(Dnew)
            else:
                if inp.wfembed.purify == "mcw":
                    Dmat[0] = McWeeny_pur(Dnew)
                if inp.wfembed.purify == "truf":
                    Dmat[0] = Truf_pur(Dnew)
                if inp.wfembed.purify == "pals":
                    Dmat[0] = Palser_pur(Dnew)
            eig_val, eig_mat = np.linalg.eig(Dnew)
            Dmat[0] = Dnew

            if inp.method == "ccsd" or inp.method == "ccsd(t)":
                Dmat[0] = (reduce (np.dot, (mCCSD.mo_coeff, np.dot(Dmat[0], mCCSD.mo_coeff.T)))) #Convert to AO from purified MO matrix
            if re.match(re.compile('cas\[.*\].*'), inp.method):
                mocas = mcSCF.mo_coeff    # Convert from MO to AO basis
                casdmao = reduce(np.dot, (mocas, Dmat[0], mocas.T))
                Dmat[0] = casdmao
            inp.timer.end("DMP")

        # get interaction energy
        inp.timer.start("INTERACTION ENERGIES")
        ldosup = not inp.embed.localize
        if ldosup and wf_cycle < 1: pstr ("Supermolecular DFT")
        Etot, EA = get_interaction_energy(inp, Dmat, Smat, supMol=(not inp.wfembed), ldosup=ldosup)
        if not Ediff:
            Ediff = inp.Esup - Etot
        inp.timer.end("INTERACTION ENERGIES")

        if inp.compden == "dft":
            pstr ("Compare Densities")
            inp.timer.start("Density Difference")
            sDmat = inp.sSCF.make_rdm1()
            mDmat_1 = np.concatenate((Dmat[0], np.zeros((Dmat[1].shape[0], Dmat[0].shape[0]))))
            mDmat_2 = np.concatenate((np.zeros((Dmat[0].shape[0], Dmat[1].shape[0])), Dmat[1]))
            mDmat = np.concatenate((mDmat_1, mDmat_2), axis=1)
            d_den = get_delta_den(inp.sSCF, sDmat, mDmat, inp.grids)
            print ("Delta Density       {0:19.14f}".format(d_den))
            inp.timer.end("Density Difference")

        wf_err = 0
        if inp.wfembed:
            if (wf_err > wf_conv_criteria):
                print ("WF CYCLES NOT CONVERGED!")
            for subsystem in range(len(Dmat)):
                wf_err += sp.linalg.norm(Dmat[subsystem] - old_Dmat[subsystem])
            print ("WF-in-DFT Cycle Errors iter:{1:3d}         {0:19.14f}".format(wf_err, wf_cycle))
            if (wf_err <= wf_conv_criteria):
               pstr("WF-in-DFT cycles converged!", delim=" ", addline=False)
            if wf_cycle >= 50:
                Dmat[0] = np.multiply(inp.wfembed.damp, old_Dmat[0]) + np.multiply((1.-inp.wfembed.damp), Dmat[0])
                Dmat[1] = np.multiply(inp.wfembed.damp, old_Dmat[1]) + np.multiply((1.-inp.wfembed.damp), Dmat[1])

        pstr ("Embedding Energies", delim="=")
        Eint = Etot - EA
        print ("Subsystem DFT              {0:19.14f}".format(EA))
        print ("Interaction                {0:19.14f}".format(Eint))
        print ("DFT-in-DFT                 {0:19.14f}".format(Etot))
        print ("Supermolecular DFT         {0:19.14f}".format(inp.Esup))
        if inp.supsmearsig:
            print ("Supmol DFT no F-Smear      {0:19.14f}".format(inp.Esup_nosmear))
        print ("Difference                 {0:19.14f}".format(Ediff))
        print ("Corrected DFT-in-DFT       {0:19.14f}".format(Etot + Ediff))
        if inp.eHI is not None:
            print ("WF-in-DFT                  {0:19.14f}".format(inp.eHI+Eint))
            print ("Corrected WF-in-DFT        {0:19.14f}".format(inp.eHI+Eint+Ediff))

        wf_cycle += 1

    if inp.gencube == "final":
        inp.timer.start("Generate Cube file")
        molA_cubename = filename.split(".")[0] + "_A.cube"
        molB_cubename = filename.split(".")[0] + "_B.cube"
        cubegen.density(inp.mol[0], molA_cubename, Dmat[0])
        cubegen.density(inp.mol[1], molB_cubename, Dmat[1])
        inp.timer.end("Generate Cube file")

    # End timing and print final wrap-up.
    inp.timer.close()
    pstr ("", delim="*")
    pstr ("END OF CALCULATION", delim="*", fill=False, addline=False)
    pstr ("", delim="*", addline=False)
    return inp

def get_input_files():
    '''Get input files from command line.'''
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys

    # Parse input files.
    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('input_files', nargs='*', default=sys.stdin,
                        help='The input files to submit.')
    args = parser.parse_args()

    for filename in args.input_files:
        inp = main(filename)

def pstr(st, delim="=", l=80, fill=True, addline=True, after=False):
    '''Print formatted string <st> to output'''
    if addline: print ("")
    if len(st) == 0:
        print (delim*l)
    elif len(st) >= l:
        print (st)
    else:
        l1 = int((l-len(st)-2)/2)
        l2 = int((l-len(st)-2)/2 + (l-len(st)-2)%2)
        if fill:
            print (delim*l1+" "+st+" "+delim*l2)
        else:
            print (delim+" "*l1+st+" "*l2+delim)
    if after: print ("")

if __name__=='__main__':
    get_input_files()
