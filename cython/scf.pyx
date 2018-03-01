from __future__ import print_function
import re
from copy import deepcopy as copy
from pyscf import scf, dft, cc
from pyscf import lib

import numpy as np
cimport numpy as np
import scipy as sp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

NTYPE = np.int
ctypedef np.int_t NTYPE_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def get_scf(inp, mol):
    '''Perform SCF on an individual subsystem.'''

    # get pySCF SCF object
    if inp.embed.dft:
        mSCF = scf.RKS(mol)
        mSCF.xc = inp.embed.method
        mSCF.small_rho_cutoff = 1e-20
    else:
        mSCF = scf.RHF(mol)

    # set some defaults
    mSCF.diis = True
    mSCF.grids = inp.grids
    mSCF.init_guess = 'atom'
    mSCF.damp = inp.damp
    mSCF.level_shift = inp.shift
    if inp.memory is not None: mSCF.max_memory = inp.memory

    # set some other defaults
    mSCF.conv_tol = inp.conv
    mSCF.conv_tol_grad = inp.grad
    mSCF.max_cycle = inp.maxiter

    # return object
    return mSCF


def get_2e_matrix(inp, Dmat):
    '''Calculates the effective potential due to both density
    matrices A and B.'''

    cdef int nS = inp.sSCF.mol.nao_nr()
    sub2sup = inp.sub2sup

    # make supermolecular density matrix
    cdef np.ndarray[DTYPE_t, ndim=2] dm = np.zeros((nS, nS))
    for i in range(inp.nsubsys):
        dm[np.ix_(sub2sup[i], sub2sup[i])] += Dmat[i]
    
    # get and return effective potential
    V = inp.sSCF.get_veff(dm=dm)
    #print ("V")
    #print (V)
    return V

def do_embedding(int A, inp, mSCF, np.ndarray[DTYPE_t, ndim=2] Hcore, Dmat,
                 np.ndarray[DTYPE_t, ndim=2] Smat, np.ndarray[DTYPE_t, ndim=2] Fock,
                 int cycles=1, DTYPE_t conv=1e-6, llast=False):
    '''Embed subsystem A in all other subsystems.'''

    # initialize
    cdef int sna = inp.smol.nao_nr()
    cdef int nA = inp.mol[A].nao_nr()
    cdef int B
    cdef np.ndarray[DTYPE_t, ndim=1] E = np.zeros((nA))
    cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros((nA, nA))
    cdef np.ndarray[DTYPE_t, ndim=2] Dold = np.zeros((nA, nA))
    cdef np.ndarray[DTYPE_t, ndim=2] Dnew = np.zeros((nA, nA))
    cdef int i
    cdef DTYPE_t error, eproj
    s2s = inp.sub2sup

    # if this is the first subsystem, reconstruct the total Fock matrix
    if A==0:
        Fock = sp.copy(Hcore)
        inp.timer.start('2e matrices')
        Fock += get_2e_matrix(inp, Dmat)
        inp.timer.end('2e matrices')

        # use DIIS on total fock matrix
        if inp.embed.diis and inp.ift >= inp.embed.diis:
            if inp.DIIS is None:
                inp.DIIS = lib.diis.DIIS()
            Fock = inp.DIIS.update(Fock)

    cdef np.ndarray[DTYPE_t, ndim=2] SAA = Smat[np.ix_(s2s[A], s2s[A])]
    cdef np.ndarray[DTYPE_t, ndim=2] POp = np.zeros((nA, nA))

    # cycle over all other subsystems
    for B in range(inp.nsubsys):
        if B==A: continue

        SAB = Smat[np.ix_(s2s[A], s2s[B])]
        SBA = Smat[np.ix_(s2s[B], s2s[A])]

        # get mu-parameter projection operator
        if inp.embed.operator.__class__ in (int, float):
            inp.timer.start('mu operator')
            POp += inp.embed.operator * np.dot( SAB, np.dot( Dmat[B], SBA ))
            inp.timer.end('mu operator')

        elif inp.embed.operator in ('huzinaga', 'huz'):
            FAB = Fock[np.ix_(s2s[A], s2s[B])]
            FDS = np.dot( FAB, np.dot( Dmat[B], SBA ))
            POp += - 0.5 * ( FDS + FDS.transpose() )

        elif inp.embed.operator in ('huzinagafermi', 'huzfermi'):
            FAB = Fock[np.ix_(s2s[A], s2s[B])]
            #The max of the fermi energy
            efermi = max(inp.Fermi[0], inp.Fermi[1])
            FAB -= SAB * efermi
            FDS = np.dot( FAB, np.dot( Dmat[B], SBA ))
            POp += - 0.5 * ( FDS + FDS.transpose() )


    # get elements of the Fock operator for this subsystem
    # and add projection operator
    cdef np.ndarray[DTYPE_t, ndim=2] FAA = Fock[np.ix_(s2s[A], s2s[A])]
    FAA += POp

    # update hcore of this subsystem (for correct energies)
    if llast or cycles > 1:
        vA = inp.mSCF[A].get_veff(dm=Dmat[A])
        hcore = FAA - vA
        inp.mSCF[A].get_hcore = lambda *args: hcore
    else:
        hcore = None

    # if we need to do a hartree-fock calculation
    if llast and inp.embed.method != inp.method:
        if (inp.method in ('hf', 'hartree-fock', 'ccsd', 'ccsd(t)', 'fci') or re.match(re.compile('cas\[.*\].*'), inp.method)):
            mSCF = scf.RHF(inp.mSCF[A].mol)
        else:
            mSCF = scf.RKS(inp.mSCF[A].mol)
            mSCF.grids = inp.sSCF.grids
            mSCF.xc = inp.method

        if(inp.embed.wfguess not in ('hf')):
            mSCF = scf.RKS(inp.mSCF[A].mol)
            mSCF.grids = inp.sSCF.grids
            mSCF.xc = inp.embed.method

        mSCF.get_hcore = lambda *args: hcore
        if inp.memory is not None: mSCF.max_memory = inp.memory
    else:
        mSCF = inp.mSCF[A]

    # do SCF cycles
    if llast and inp.embed.method != inp.method:
        mSCF, Dnew, error , inp.Fermi[A] = diagonalize(mSCF, Dmat[A], SAA,
            cycles, conv, Fock=None, hcore=hcore, sigma=inp.subsystem[A].smearsig)
    else:
        mSCF, Dnew, error, inp.Fermi[A] = diagonalize(mSCF, Dmat[A], SAA,
            cycles, conv, Fock=FAA, hcore=hcore, sigma=inp.subsystem[A].smearsig)

    # for the last run, do 1 call to the pySCF kernel to get some values
    if llast:
        mSCF.max_cycle = inp.maxiter
        mSCF.kernel(dm0=Dnew)
        Dnew = mSCF.make_rdm1()
        mSCF.max_cycle = inp.maxiter
        #if inp.embed.method != inp.method:
        #    inp.mSCF[A].max_cycle = 1 
        #    inp.mSCF[A].kernel(dm0=Dmat[A])
        #    Dmat[A] = inp.mSCF[A].make_rdm1()
        #    inp.mSCF[A].max_cycle = inp.embed.subcycles

    # trace of projection operator with density
    eproj = np.trace(np.dot(POp, Dnew))

    # return new density matrix
    return Fock, mSCF, Dnew, eproj, error

def do_final_dft_embed(inp, mSCF, Hcore, Dmat, Smat, Fock, cycles=100, conv=1e-6, A=1):

    '''Perform a final DFT using the density determined by the WF method.'''
    # initialize
    cdef int sna = inp.smol.nao_nr()
    cdef int nA = inp.mol[A].nao_nr()
    cdef int B
    cdef np.ndarray[DTYPE_t, ndim=1] E = np.zeros((nA))
    cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros((nA, nA))
    cdef np.ndarray[DTYPE_t, ndim=2] Dold = np.zeros((nA, nA))
    cdef np.ndarray[DTYPE_t, ndim=2] Dnew = np.zeros((nA, nA))
    cdef int i
    cdef DTYPE_t error, eproj
    s2s = inp.sub2sup

    #Reconstruct the total Fock matrix
    Fock = sp.copy(Hcore)
    inp.timer.start('2e matrices')
    Fock += get_2e_matrix(inp, Dmat)
    inp.timer.end('2e matrices')

    # use DIIS on total fock matrix
    if inp.embed.diis and inp.ift >= inp.embed.diis:
        if inp.DIIS is None:
            inp.DIIS = lib.diis.DIIS()
        Fock = inp.DIIS.update(Fock)

    cdef np.ndarray[DTYPE_t, ndim=2] SAA = Smat[np.ix_(s2s[A], s2s[A])]
    cdef np.ndarray[DTYPE_t, ndim=2] POp = np.zeros((nA, nA))

    # cycle over all other subsystems
    for B in range(inp.nsubsys):
        if B==A: continue

        SAB = Smat[np.ix_(s2s[A], s2s[B])]
        SBA = Smat[np.ix_(s2s[B], s2s[A])]

        # get mu-parameter projection operator
        if inp.embed.operator.__class__ in (int, float):
            inp.timer.start('mu operator')
            POp += inp.embed.operator * np.dot( SAB, np.dot( Dmat[B], SBA ))
            inp.timer.end('mu operator')

        elif inp.embed.operator in ('huzinaga', 'huz'):
            FAB = Fock[np.ix_(s2s[A], s2s[B])]
            FDS = np.dot( FAB, np.dot( Dmat[B], SBA ))
            POp += - 0.5 * ( FDS + FDS.transpose() )

    # get elements of the Fock operator for this subsystem
    # and add projection operator
    cdef np.ndarray[DTYPE_t, ndim=2] FAA = Fock[np.ix_(s2s[A], s2s[A])]
    FAA += POp

    # update hcore of this subsystem (for correct energies)
    vA = inp.mSCF[A].get_veff(dm=Dmat[A])
    hcore = FAA - vA
    inp.mSCF[A].get_hcore = lambda *args: hcore

    # if we need to do a hartree-fock calculation
    mSCF = scf.RKS(inp.mSCF[A].mol)
    mSCF.xc = inp.embed.method
    mSCF.grids = inp.sSCF.grids
    mSCF.get_hcore = lambda *args: hcore
    if inp.memory is not None: mSCF.max_memory = inp.memory

    # do SCF cycles
    mSCF, Dnew, error, inp.Fermi[A] = diagonalize(mSCF, Dmat[A], SAA,
        cycles, conv, Fock=FAA, hcore=hcore, sigma=inp.subsystem[A].smearsig)

    # trace of projection operator with density
    eproj = np.trace(np.dot(POp, Dnew))

    return Fock, mSCF, Dnew, eproj, error


def do_supermol_scf(inp, Dmat, Smat):

    cdef int nS = inp.sSCF.mol.nao_nr()
    s2s = inp.sub2sup

    # make supermolecular density matrix
    cdef np.ndarray[DTYPE_t, ndim=2] dm = np.zeros((nS, nS))
    if Dmat.__class__ is list:
        for i in range(inp.nsubsys):
            dm[np.ix_(s2s[i], s2s[i])] += Dmat[i]
    else:
        dm = np.copy(Dmat)

    hcore = inp.sSCF.get_hcore()
    inp.sSCF, dnew, error, fermi = diagonalize(inp.sSCF, dm, Smat, inp.maxiter, inp.conv, hcore=hcore, sigma=inp.supsmearsig)
    inp.sSCF.kernel(dm0=dnew)
    dm_nosmear = inp.sSCF.make_rdm1()
    inp.Esup_nosmear = inp.sSCF.energy_tot(dm=dm_nosmear)
    inp.sSCF, dnew, error, fermi = diagonalize(inp.sSCF, dm_nosmear, Smat, inp.maxiter, inp.conv, hcore=hcore, sigma=inp.supsmearsig)
    inp.Esup = inp.sSCF.energy_tot(dm=dnew)
    print ("Supermolecular DFT   {0:19.14f}".format(inp.Esup))

    return inp.sSCF

def diagonalize(mSCF, Dmat, Smat, int cycles, float conv, Fock=None, hcore=None, sigma=None):

    Dold = sp.copy(Dmat)
    cdef int nA = Dmat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] N = np.zeros((nA))
    cdef np.ndarray[DTYPE_t, ndim=1] E = np.zeros((nA))
    cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros((nA, nA))
    N[:mSCF.mol.nelectron//2] = 2.

    DIIS = lib.diis.DIIS()

    cdef int i = 0
    cdef float error = 1.
    while (i < cycles) and (error > conv):
        i += 1

        # update fock matrix
        if Fock is None or i > 1:
            vA = mSCF.get_veff(dm=Dold)
            Fock = hcore + vA
            Fock = DIIS.update(Fock)


        # diagonalize
        if i==2: print ('  subiter:   1      |ddm|: {0:12.6e}'.format(error))
        E, C = sp.linalg.eigh(Fock, Smat)
        Dnew = np.dot( C * N, C.transpose() )
        error = sp.linalg.norm(Dold - Dnew)
        if i>1: print ('  subiter: {0:>3d}      |ddm|: {1:12.6e}'.format(i, error))
        Dold = np.copy(Dnew)

        # get fermi energy (average of HOMO and LUMO)
        norbs = (mSCF.mol.nelectron) // 2
        e_sorted = np.sort(E)
        fermi = (e_sorted[norbs] + e_sorted[norbs-1]) / 2.

        # get molecular occupation
        if sigma is None:
            mo_occ = np.zeros_like(E)
            mo_occ[E<fermi] = 2.
        else:
            mo_occ = ( E - fermi ) / sigma
            ie = np.where( mo_occ < 1000 )
            i0 = np.where( mo_occ >= 1000 )
            mo_occ[ie] = 2. / ( np.exp( mo_occ[ie] ) + 1. )
            mo_occ[i0] = 0.

        Dnew = np.dot(C * mo_occ, C.transpose().conjugate())

    return mSCF, Dnew, error, fermi

