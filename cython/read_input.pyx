#!/usr/bin/env python
from __future__ import print_function, division

import input_reader
from pyscf import gto
import h5py
import sys
import numpy as np

from cpython cimport array
import array



def read_input(filename):
    '''Reads a formatted pySCF input file, and generates
    a relevant LIST of pySCF Mole objects from it.

    Input: <filename> the filename of the pySCF input file
    Output: a LIST of pySCF Mole objects.
    '''

    # initialize reader for a pySCF input
    reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=False)

    # add subsystems block
    subsys = reader.add_block_key('subsystem', required=True, repeat=True)
    subsys.add_regex_line('atom',
        '\s*([A-Za-z.]+)\s+(\-?\d\.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)', repeat=True)
    subsys.add_line_key('charge', type=int)     # local subsys charge
    subsys.add_line_key('spin', type=int)       # local subsys spin
    subsys.add_line_key('basis')                # local subsys basis
    subsys.add_line_key('smearsig', type=float, default=None)     # local subsys smearing
    subsys.add_line_key('unit', type=('angstrom','a','bohr','b')) # local subsys unit

    # add WF embedding block
    WFembed = reader.add_block_key('wfembed', required=False, default=None)
    WFembed.add_line_key('cycles', type=int, default=0)           # max freeze-and-thaw cycles
    WFembed.add_line_key('conv', type=float, default=None)        # f&t conv tolerance
    WFembed.add_line_key('diis', type=int, default=1)             # start DIIS (0 to turn off)
    WFembed.add_line_key('purify', type=str, default='mcw')       # purification method
    WFembed.add_line_key('damp', type=float, default=0.)          # WF density damping parameter
    WFembed.add_boolean_key('natorb', action='natorb')            # use natural orbitals

    # add embedding block
    embed = reader.add_block_key('embed', required=True)
    embed.add_line_key('cycles', type=int, default=0)           # max freeze-and-thaw cycles
    embed.add_line_key('conv', type=float, default=None)        # f&t conv tolerance
    embed.add_line_key('method', type=str)                      # embedding method
    embed.add_line_key('wfguess', type=str, default='hf')                      # embedding method
    embed.add_line_key('diis', type=int, default=1)             # start DIIS (0 to turn off)
    embed.add_boolean_key('localize')                           # whether to localize orbitals
    embed.add_boolean_key('freezeb')                            # optimize only subsystem A
    embed.add_line_key('subcycles', type=int, default=1)        # number of subsys diagonalizations
    operator = embed.add_mutually_exclusive_group(dest='operator', required=True)
    operator.add_line_key('mu', type=float, default=1e6)        # manby operator by mu
    operator.add_boolean_key('manby', action=1e6)               # manby operator
    operator.add_boolean_key('huzinaga', action='huzinaga')     # huzinaga operator
    operator.add_boolean_key('hm', action='hm')                 # modified huzinaga
    operator.add_boolean_key('huzinagafermi', action='huzinagafermi')     # huzinaga-fermi shifted operator
    operator.add_boolean_key('huzfermi', action='huzfermi')                 # modified huzinaga-fermi shifted operator

    # add simple line keys
    reader.add_line_key('memory', type=(int, float))            # max memory in MB
    reader.add_line_key('unit', type=('angstrom','a','bohr','b')) # global coord unit
    reader.add_line_key('basis', default='sto-3g')              # global basis
    reader.add_line_key('method', default='hf')                 # SCF method
    reader.add_line_key('conv', type=float, default=1e-8)       # conv_tol
    reader.add_line_key('grad', type=float, default=1e-4)       # conv_tol_grad
    reader.add_line_key('maxiter', type=int, default=50)        # max SCF iterations
    reader.add_line_key('grid', type=int, default=3)            # becke integration grid
    reader.add_line_key('verbose', type=int, default=4)         # pySCF verbose level
    reader.add_line_key('damp', type=float, default=0.)         # SCF damping parameter
    reader.add_line_key('shift', type=float, default=0.)        # SCF level-shift parameter
    reader.add_line_key('supsmearsig', type=float, default=None)   # supermolecular dft smearing
    reader.add_line_key('save')                                 # save Dmats objects to file
    reader.add_boolean_key('readchk')                                 # read Dmats from file
    reader.add_line_key('gencube', default="None")              # generate Cube file
    reader.add_line_key('compden', default="None")              # compare densities

    # add simple boolean keys
    reader.add_boolean_key('analysis')                          # whether to do pySCF analysis
    reader.add_boolean_key('debug')                             # debug flag
    # read the input filename
    inp  = reader.read_input(filename)
    inp.filename = filename # save filename for later

    # some defaults
    if inp.embed.method is None: inp.embed.method = inp.method
    if inp.embed.method in ('ccsd', 'ccsd(t)', 'fci'): inp.embed.method = 'hf'

    # check whether this is a DFT calculation or not
    inp.dft = True
    if inp.method in ('hf', 'hartree-fock', 'ccsd', 'ccsd(t)', 'fci'): inp.dft = False
    inp.embed.dft = True
    if inp.embed.method in ('hf', 'hartree-fock', 'ccsd', 'ccsd(t)', 'fci'): inp.embed.dft = False
    inp.embed.e_dft = True

    # sanity checks
    if len(inp.subsystem) < 2: sys.exit("Cannot perform subsystem calculation with one subsystem!")
    if inp.embed.conv is None: inp.embed.conv = inp.conv
    if inp.embed.operator.__class__ is float:
        if inp.embed.conv < (1. / inp.embed.operator):
            print ("Cannot converge freeze-and-thaw to better than 1/mu !")
            inp.embed.conv = max(1. / inp.embed.operator, inp.conv)
            print ("Setting EMBED/CONV to {0:7.1e}".format(inp.embed.conv))
    elif inp.conv > inp.embed.conv:
        print ("EMBED/CONV should be >= GLOBAL/CONV!")
        inp.embed.conv = inp.conv
        print ("Setting EMBED/CONV to {0:7.1e}".format(inp.embed.conv))


    # initialze pySCF molecule object list and cycle
    mols = [None for i in range(len(inp.subsystem))]
    cdef int nghost
    for m in range(len(inp.subsystem)):
        mol = gto.Mole()

        # collect atoms in pyscf format
        mol.atom = []
        ghbasis = []
        for r in inp.subsystem[m].atom:
            if 'ghost.' in r.group(1).lower() or 'gh.' in r.group(1).lower():
                ghbasis.append(r.group(1).split('.')[1])
                rgrp1 = 'ghost:{0}'.format(len(ghbasis))
                mol.atom.append([rgrp1, (float(r.group(2)), float(r.group(3)), float(r.group(4)))])
            else:
                mol.atom.append([r.group(1), (float(r.group(2)), float(r.group(3)), float(r.group(4)))])

        # build dict of basis for each atom
        mol.basis = {}
        nghost = 0
        subbas = [inp.subsystem[m].basis if inp.subsystem[m].basis else inp.basis][0]
        for i in range(len(mol.atom)):
            if 'ghost' in mol.atom[i][0]:
                mol.basis.update({mol.atom[i][0]: gto.basis.load(subbas, ghbasis[nghost])})
                nghost += 1
            else:
                mol.basis.update({mol.atom[i][0]: gto.basis.load(subbas, mol.atom[i][0])})

        # use local values first, then global values, if they exist
        if inp.memory is not None: mol.max_memory = inp.memory
        if inp.subsystem[m].unit is not None:
            mol.unit = inp.subsystem[m].unit
        elif inp.unit is not None:
            mol.unit = inp.unit
        if inp.subsystem[m].charge is not None: mol.charge = inp.subsystem[m].charge
        if inp.subsystem[m].spin is not None: mol.spin = inp.subsystem[m].spin
        mol.verbose = inp.verbose
        mol.build(dump_input=False)

        # add to list mols
        mols[m] = mol

    # replace subsystem with mols object list
    #del(inp.subsystem)
    inp.mol = mols[:] #This used to copy mols but I don't think copy is necessary
    inp.nsubsys = len(inp.mol)


  #  #Here it is!
  #  if 'fermi' in inp.h5py:
  #      inp.Fermi = inp.h5py['fermi']

  #  else:
  #      inp.Fermi = np.zeros((inp.nsubsys))
  #      inp.Fermi = inp.h5py.create_dataset('fermi', data=inp.Fermi)

    return inp
