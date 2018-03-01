"""Huzinaga Embedding Helper module

This module provides helper functions for Huzinaga embedding method.
"""

from scf import do_supermol_scf
from pyscf import gto, scf, dft
from copy import deepcopy as copy
import re

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

NTYPE = np.int32
ctypedef np.int32_t NTYPE_t

LTYPE = np.int
ctypedef np.int_t LTYPE_t


cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def dist2(c1, c2):
    """Returns the distance squared.

    Parameters
    ----------
    c1 : numpy vector
        Vector 1
    c2 : numpy vector
        Vector 2

    Returns
    -------
    float
        The distance between two vectors squared.
    """
    return np.dot(c1-c2,c1-c2)

@cython.boundscheck(False)
@cython.wraparound(False)
def concatenate_mols(mA, mB, ghost=False):
    '''Takes two pySCF mol objects, and concatenates them.  If the option "ghost" is given, it keeps the ghost atoms
    from each mol, otherwise only the "real" atoms are copied.'''

    mC = gto.Mole()

    atmC = []
    ghstatm = {}
    cdef int nghost, oghost, i 
    nghost = 0

    # copy all atoms from mole A
    for i in range(mA.natm):
        if 'ghost' in mA.atom_symbol(i).lower():
            if ghost:
                nghost += 1
                atmC.append([mA.atom_symbol(i), mA.atom_coord(i)])
        else:
            atmC.append([mA.atom_symbol(i), mA.atom_coord(i)])
    mC.basis = mA.basis.copy()

    # copy all atoms from mole B
    for i in range(mB.natm):
        if 'ghost' in mB.atom_symbol(i).lower():
            if ghost:
                nghost += 1
                oghost = int(mB.atom_symbol(i).split(':')[1])
                newsym = 'GHOST:{0}'.format(nghost)
                atmC.append([newsym, mB.atom_coord(i)])
                mC.basis.update({newsym.lower(): mB.basis['ghost:{0}'.format(oghost)]})
        else:
            atmC.append([mB.atom_symbol(i), mB.atom_coord(i)])
            mC.basis.update({mB.atom_symbol(i): mB.basis[mB.atom_symbol(i)]})

    mC.atom = atmC
    mC.verbose = mA.verbose
    mC.charge = mA.charge + mB.charge
    mC.unit = 'bohr' # atom_coord is always stored in bohr (?)
    mC.build(dump_input=False)

    return mC

@cython.boundscheck(False)
@cython.wraparound(False)
def gen_grids(inp):

    # generate a grid based on concatenated subsystems
    mC = concatenate_mols(inp.mol[0], inp.mol[1], ghost=False)
    for i in range(2,inp.nsubsys):
        mC = concatenate_mols(mC, inp.mol[i], ghost=False)
    inp.smol = mC
    grids = dft.gen_grid.Grids(mC)
    grids.level = inp.grid
    grids.build()

    return grids

def get_delta_den(mSCF, Dmat1, Dmat2, grid):
    """Get the delta density of two different matrices"""
    gAO = dft.numint.eval_ao(mSCF.mol, grid.coords)
    rho = np.sum(gAO * np.dot(gAO, np.subtract(Dmat1, Dmat2)), axis=1)
    return np.sum(np.multiply(grid.weights, np.square(rho)))
    
    
def get_interaction_energy(inp, Dmat, Smat, supMol=False, ldosup=False):

    cdef int nS = inp.sSCF.mol.nao_nr()
    s2s = inp.sub2sup

    # make supermolecular density matrix
    cdef np.ndarray[DTYPE_t, ndim=2] dm = np.zeros((nS, nS))
    for i in range(inp.nsubsys):
        dm[np.ix_(s2s[i], s2s[i])] += Dmat[i]

    # get total energy
    V = inp.sSCF.get_veff(dm=dm)
    Etot = inp.sSCF.energy_tot(dm=dm)

    # get energy of subsystem A
    VA = inp.mSCF[0].get_veff(dm=Dmat[0])
    EA = inp.mSCF[0].energy_tot(dm=Dmat[0])

    # do supermolecular calculation if needed
    if ldosup and supMol:
        inp.sSCF = do_supermol_scf(inp, dm, Smat)

    return Etot, EA
