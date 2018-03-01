#A module which enables DFT embedding geometry optimization.
#By DanG
from __future__ import print_function, division
from read_input import read_input
from pyscf import cc, gto, scf
from pyscf.tools import cubegen
from copy import deepcopy as copy
import simple_timer
import pickle
import gzip
import os
from integrals import get_interaction_energy, concatenate_mols
from integrals import gen_grids
from scf import get_scf
from scf import do_embedding
from scf import do_supermol_scf
from scf import do_final_dft_embed
from localize import get_sub2sup
from pyscf import grad
import sys

import numpy as np
import scipy as sp
import rmsd

def dist(coord1, coord2):
    tot = 0
    for i in range(len(coord1)):
        tot += (coord1[i] - coord2[i]) ** 2.
    return (tot ** 0.5)

## def shift_barycenter(molecule):
##     c = gto.mass_center(molecule._atom)
##     new_atom = ""
##     for i in range(len(molecule._atom)):
##         new_atom += molecule._atom[i][0] + " "
##         for j in range(len(molecule._atom[i][1])):
##             new_atom += str(molecule._atom[i][1][j] - c[j]) + "  "
##         new_atom += "; "
## 
##     molecule.atom = new_atom
##     molecule.build()
##     return molecule
## 
## def compute_correlation_matrix(molecule1, molecule2):
## 
##     corr_matrix = np.zeros((3,3))
##     for i in range(len(molecule1._atom)):
##         for j in range(3):
##             for k in range(2, -1, -1):
##                 corr_matrix[j][k] += (molecule1._atom[i][1][j] * molecule2._atom[i][1][k])
## 
##     return corr_matrix
## 
## def compute_f_matrix(corr_matrix):
## 
##     f_matrix = np.zeros((4,4))
##     f_matrix[0][0] = np.trace(corr_matrix)
##     f_matrix[1][1] = 2 * corr_matrix[0][0] - np.trace(corr_matrix)
##     f_matrix[2][2] = 2 * corr_matrix[1][1] - np.trace(corr_matrix)
##     f_matrix[3][3] = 2 * corr_matrix[2][2] - np.trace(corr_matrix)
## 
##     f_matrix[0][1] = f_matrix[1][0] = corr_matrix[1][2] - corr_matrix[2][1]
##     f_matrix[0][2] = f_matrix[2][0] = corr_matrix[2][0] - corr_matrix[0][2]
##     f_matrix[0][3] = f_matrix[3][0] = corr_matrix[0][1] - corr_matrix[1][0]
## 
##     
##     f_matrix[1][2] = f_matrix[2][1] = corr_matrix[0][1] + corr_matrix[1][0]
##     f_matrix[1][3] = f_matrix[3][1] = corr_matrix[0][2] + corr_matrix[2][0]
## 
##     f_matrix[2][3] = f_matrix[3][2] = corr_matrix[1][2] + corr_matrix[2][1]
## 
##     return f_matrix
## 
## def compute_RMSD(molecule1, molecule2):
## 
##     #Step 1. Shift both molecules to their barycenters.
##     mol1 = shift_barycenter(molecule1)
##     mol2 = shift_barycenter(molecule2)
## 
##     #Step 2. Calculate the RMSD at those locations
##     corr_matrix = compute_correlation_matrix(mol1, mol2)
##     f_matrix = compute_f_matrix(corr_matrix)
##     lam_vals, g_vals = np.linalg.eigh(f_matrix)
##     lam_max = lam_vals[-1]
##     g_max = g_vals[-1]
## 
##     e_val = calc_RMSD(molecule1, molecule2, lam_max)
##     print e_val
##      
## 
## 
## def calc_RMSD(mol1, mol2, lam):
## 
##     molSum = np.zeros(3)
##     numMol = 0.
##     for i in range(len(mol1._atom)):
##         numMol += 1.
##         molSum += np.square(np.linalg.norm(mol1._atom[i][1])) + np.square(np.linalg.norm(mol2._atom[i][1]))
## 
##     rmsd_val = np.sqrt(np.divide(np.subtract(molSum[0], 2 * lam), numMol))
##     return rmsd_val
##     
## def RMSD_gradient():
##     pass

def mol_to_coord_mat(molecule1):
    mat = np.zeros((molecule1.natm, 3))
    for i in range(molecule1.natm):
        for j in range(3):
            mat[i][j] = molecule1._atom[i][1][j]

    return mat


def calc_rmsd(mol1, mol2):
    mol_coord1 = mol_to_coord_mat(mol1)
    mol_coord2 = mol_to_coord_mat(mol2)
    
    #Localize on centroid
    mol_coord1 -= rmsd.centroid(mol_coord1)
    mol_coord2 -= rmsd.centroid(mol_coord2)

    return rmsd.quaternion_rmsd(mol_coord1, mol_coord2)
    
#Calls the FT code using the input format and outputs an optimized geometry
def main(filename):
    #uses the parameters specified in the inp object to optimize the geometry of the given molecule and outputs a file with the updated geometry

    #TO PARAMATERIZE
    min_grad = (1 * 10^(-6))
         
    # initialize and print header
    pstr ("", delim="*", addline=False)
    pstr (" FREEZE-AND-THAW GEOMETRY CALCULATION ", delim="*", fill=False, addline=False)
    pstr ("", delim="*", addline=False)

    # print input options to stdout
    pstr ("Input File")
    [print (i[:-1]) for i in open(filename).readlines() if ((i[0] not in ['#', '!'])
        and (i[0:2] not in ['::', '//']))]
    pstr ("End Input", addline=False)

    # read input file
    inp = read_input(filename)
    inp.timer = simple_timer.timer()
    nsub = inp.nsubsys
    inp.DIIS = None
    inp.ndiags = [0 for i in range(nsub)]

    # use supermolecular grid for all subsystem grids
    inp.grids = gen_grids(inp)
    sna = inp.smol.nao_nr()
    na = [inp.mol[i].nao_nr() for i in range(nsub)]

    # initialize 1e matrices from supermolecular system
    inp.timer.start("OVERLAP INTEGRALS")
    pstr ("Getting subsystem overlap integrals", delim=" ")
    inp.sSCF = get_scf(inp, inp.smol)
    Hcore = inp.sSCF.get_hcore()
    Smat = inp.sSCF.get_ovlp()
    Fock = np.zeros((sna, sna))
    Dmat = [None for i in range(nsub)]
    inp.timer.end("OVERLAP INTEGRALS")

    # get initial density for each subsystem
    inp.timer.start("INITIAL SUBSYSTEMS")
    inp.mSCF = [None for i in range(nsub)] # SCF objects
    pstr ('Initial subsystem densities', delim=' ')
    for i in range(nsub):
        inp.mSCF[i] = get_scf(inp, inp.mol[i])

        # read density from file
        if inp.read and os.path.isfile(inp.read+'.dm{0}'.format(i)):
            print ("Reading initial densities from file: {0}".format(inp.read+'.dm{0}'.format(i)))
            Dmat[i] = pickle.load(open(inp.read+'.dm{0}'.format(i), 'rb'))

        # guess density
        else:
            print ("Guessing initial subsystem densities")
            Dmat[i] = inp.mSCF[i].init_guess_by_atom()
            #Dmat[i] = inp.mSCF[i].init_guess_by_minao()
    inp.sub2sup = get_sub2sup(inp, inp.mSCF)
    inp.timer.end("INITIAL SUBSYSTEMS")

    
    # start with supermolecule calculation
    if inp.embed.localize:
        pstr ("Supermolecular DFT")
        inp.sSCF = do_supermol_scf(inp, Dmat, Smat)
        from localize import localize
        pstr ("Localizing Orbitals", delim="=")
        Dmat = localize (inp, inp.sSCF, inp.mSCF, Dmat, Smat)

    # do freeze-and-thaw cycles
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

        # cycle over active subsystems
        for a in range(nsub):
            if inp.embed.freezeb and a > 0: continue

            # perform embedding of this subsystem
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
            print ("FOCK")
            print (Fock)

    # print whether freeze-and-thaw has converged
    if ((inp.embed.cycles > 1 and inp.error < inp.embed.conv)
       or (inp.embed.cycles == 1 and err < inp.embed.conv)):
        pstr (smethod+" converged!", delim=" ", addline=False)
    elif inp.embed.cycles == 0:
        pass
    else:
        pstr (smethod+" NOT converged!", delim=" ", addline=False)

    #DFT in DFT Breakdown
    superMol = concatenate_mols(inp.mol[0], inp.mol[1])

    # do final embedded cycle at high level of theory
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

    # if WFT-in-(DFT/HF), do WFT theory calculation here
    if inp.method == 'ccsd' or inp.method == 'ccsd(t)':
        inp.timer.start("CCSD")
        pstr ("Doing CCSD Calculation on Subsystem 1")
        mCCSD = cc.CCSD(mSCF_A)
        ecc, t1, t2 = mCCSD.kernel()
        inp.eHI = mSCF_A.e_tot + ecc
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
        inp.timer.end("CCSD(T)")

        print ("Hartree-Fock Energy:  {0:19.14f}".format(mSCF_A.e_tot))
        print ("CCSD(T) Correlation:     {0:19.14f}".format(ecc))
        print ("CCSD(T) Energy:          {0:19.14f}".format(mSCF_A.e_tot+ecc))

    if inp.method == 'fci':
        from pyscf import fci
        inp.timer.start("FCI")
        pstr ("Doing FCI Calculation on Subsystem 1")
        #Found here: https://github.com/sunqm/pyscf/blob/master/examples/fci/00-simple_fci.py
        cisolver = fci.FCI(inp.mol[0], mSCF_A.mo_coeff)
        efci = cisolver.kernel()[0]
        inp.eHI = efci
        inp.timer.end("FCI")

        print ("Hartree-Fock Energy:  {0:19.14f}".format(mSCF_A.e_tot))
        print ("FCI Correlation:     {0:19.14f}".format(efci - mSCF_A.e_tot))
        print ("FCI Energy:          {0:19.14f}".format(efci))

    # get interaction energy
    inp.timer.start("INTERACTION ENERGIES")
    ldosup = not inp.embed.localize
    if ldosup: pstr ("Supermolecular DFT")
    Etot, EA = get_interaction_energy(inp, Dmat, Smat, ldosup=ldosup)
    inp.timer.end("INTERACTION ENERGIES")

    if inp.gencube == "final":
        inp.timer.start("Generate Cube file")
        molA_cubename = filename.split(".")[0] + "_A.cube"
        molB_cubename = filename.split(".")[0] + "_B.cube"
        cubegen.density(inp.mol[0], molA_cubename, Dmat[0])
        cubegen.density(inp.mol[1], molB_cubename, Dmat[1])

        inp.timer.end("Generate Cube file")

    pstr ("Embedding Energies", delim="=")
    Eint = Etot - EA
    Ediff = inp.Esup - Etot

    print ("Subsystem DFT              {0:19.14f}".format(EA))
    print ("Interaction                {0:19.14f}".format(Eint))
    print ("DFT-in-DFT                 {0:19.14f}".format(Etot))
    print ("Supermolecular DFT         {0:19.14f}".format(inp.Esup))
    print ("Difference                 {0:19.14f}".format(Ediff))
    print ("Corrected DFT-in-DFT       {0:19.14f}".format(Etot + Ediff))
    if inp.eHI is not None:
        print ("WF-in-DFT                  {0:19.14f}".format(inp.eHI+Eint))
        print ("Corrected WF-in-DFT        {0:19.14f}".format(inp.eHI+Eint+Ediff))

    # close and print timings
    inp.timer.close()

    # print end of file
    pstr ("", delim="*")
    pstr ("END OF CYCLE", delim="*", fill=False, addline=False)
    pstr ("", delim="*", addline=False)

    G_tot = grad.rks.Gradient(inp.sSCF)
    G_tot_mat = G_tot.grad()
    print (G_tot_mat)

def get_input_files():
    '''Get input files from command line.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys

    # parse input files
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

def test():
    mol1 = gto.M(
        atom = 'H 0 1 -1; H 0 1 1; O 0 0 0',
        basis = 'sto3g',
        unit = 'Bohr')

    mol1.build()

    mol2 = gto.M(
        atom = 'H 1 -1 -1; H 1 -1 1; O 1 0 0',
        basis = 'sto3g',
        unit = 'Bohr')
    mol2.build()

#    print compare_bond_distances(mol1, mol2)
#    print compare_bond_angles(mol1, mol2)

    print (calc_rmsd(mol1, mol2))

if __name__=='__main__':
    main(sys.argv[1])
