#!/volatile/home/ts234112/INFO/CONDA/miniforge3/envs/envts2/bin/python3.12
# -*- coding: utf-8 -*-

"""
KineCluE - Kinetic Cluster Expansion
T. Schuler, L. Messina, M. Nastar
kineclue_num.py (numerical calculation of transport coefficients)
!! This code requires first an analytical calculation with kineclue_main.py !!

Copyright 2018 CEA, École Nationale Supérieure des Mines de Saint-Étienne

This file is part of KineCluE.
KineCluE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
KineCluE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License
along with KineCluE. If not, see <http://www.gnu.org/licenses/>.
"""

import _pickle as pickle
import sympy as sym
import time as tm
import numpy as np
import random as rd
import mpmath as mp
import os
import copy
import sys
import logging
import datetime
from psutil import Process
import warnings as warn
from scipy import sparse
from shutil import copyfile
from itertools import permutations, product
from sympy.tensor.array import Array
from kinepy import tol, change_base, flat_list, are_equal_arrays, evalinput, trans_and_check_unicity, rotate_matrix, \
vect2deftrans, apply_symmetry_operations, tol_pbc, date, version, recursionlimit, sensitivity_study, Subconfiguration, \
check_connectivity, Configuration, print_license_notice, produce_error_and_quit, convergence_analysis, batchcreatefunc, \
solve_prec, solve_def, solve_stationary_state, solve_stationary_prec, solve_neumann, solve_def_numissues
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator
from matplotlib.backends.backend_pdf import PdfPages

def numrun(myinput: str, stream_log: bool=True):

    sys.setrecursionlimit(recursionlimit) # for Pickle, else it causes error to save important data
    start_time = tm.time() # Start measuring execution time
    np.set_printoptions(precision=15)
    process_for_memory_tracking = Process(os.getpid())  # Save process id for memory usage tracking
    chol = False  # if true uses cholesky decomposition / if false uses LU factorization
    neumann = False  # if true, uses neumann series to compute the inverse of T
    neumann_approx = [4, 2]  # the first number is the maximum power in the series, the second is the alpha parameter to improve convergence

    # Reading input file
    if os.path.exists(myinput):
        input_string = open(myinput, "r").read()
    else:
        produce_error_and_quit("ERROR! Numerical input file {} not found.".format(myinput))

    # Remove #-comments from input file
    input_string = "\n".join([string.split(sep='#')[0] for string in input_string.split(sep='\n') if len(string.split(sep='#')[0]) > 0])

    # Definition of input_dataset dictionary
    keywords = ['directory', 'precision', 'lattparam', 'temperatures', 'sensitivity', 'outoptions', 'output',
                'units', 'prefactor', 'mob', 'kraactivation', 'random', 'batch', 'interactionmodel', 'numstrain',
                'kiraloop', 'batchcreate', 'randerror', 'freqcheck', 'ndigits', 'refdisso']
    input_dataset = {key: None for key in keywords}  # dictionary for reading user input
    # Split input in keywords
    input_list = input_string.split(sep='& ')
    del input_list[0]  # delete all comments before first keyword
    # Save input data into input_dataset
    for ipt in input_list:
        keyword = ipt.split()[0].lower()  # get keyword
        if keyword == "kraactivation" or keyword == "batchcreate":
            input_dataset[keyword] = [i.split() for i in ipt.split('\n') if len(i) > 0][1:]
        else:
            input_dataset[keyword] = [i for i in ipt.replace("\n"," ").split()[1:]]  # split all entries of each keyword in a list
    del input_string, keyword, keywords, input_list, ipt

    # Directory where the results are written
    if input_dataset['directory'] is None:
        dir = './CALC/'
    elif input_dataset['directory'] == ["cwd"]:
        dir = os.getcwd() + "/"
    else:
        dir = input_dataset['directory'][0]
        # Add a slash to the end of the directory path if it's missing
        if input_dataset['directory'][0][-1] != "/":
            dir += "/"
    # Stop execution if directory does not exist
    if not os.path.exists(dir):
        produce_error_and_quit("Directory {} not found.".format(dir))

    # Setting up logfile
    logger = logging.getLogger('kinecluelog')
    logger.setLevel(logging.INFO)
    logger.info('Working in directory:  {}'.format(dir))
    fh = logging.FileHandler(dir + '/kineclue_num.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    if stream_log:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    logger.info(' __________________________________________________')
    logger.info(' |                                                |')
    logger.info(' |           KineCluE v{} - {}           |'.format(version, date))
    logger.info(' |        T. Schuler, L. Messina, M. Nastar       |')
    logger.info(' |________________________________________________|')
    logger.info('')
    print_license_notice()
    logger.info('')
    logger.info('Calculation date: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    logger.info('Working in directory:  {}'.format(dir))
    logger.info("Read numerical input file (starts at {:.3f} s)".format(tm.time()-start_time))

    # Check that required information is in the input
    if input_dataset['kraactivation'] is None:
        produce_error_and_quit("!! You MUST provide a KRAACTIVATION keyword.")
    if input_dataset['temperatures'] is None:
        produce_error_and_quit("!! You MUST provide a TEMPERATURES keyword.")

    # Load analytical results
    logger.info("Reading output file from analytical results (starts at {:.3f} s)".format(tm.time()-start_time))
    if os.path.exists(dir + "analytical_kineclue_output.pkl"):  # normal cluster calculation
        with sym.evaluate(False):
            Tcoords, Tsparse, Tdiag, L0mat, Klambda, zfunc, dissocoeff, ThRa, complist, crystal_name, all_defect_list, \
            bulkspec, uniqueconf, uniquefreq, maxkira, bulkspecies_list, bulkspecies_symdefects, dbdist, SiteCorr, \
            disfmt, fnfmt, transmat, brokenjumps, jumpmechdisp = pickle.load(open(dir+'analytical_kineclue_output.pkl', 'rb'))
    else:
        produce_error_and_quit("!! Analytical kineclue output file not found in directory {}. You need to run a symbolic calculation first with kineclue_main.".format(dir))

    # Defining working precision (if specified in input)

    if input_dataset.get('precision') is not None:
        mp.mp.dps = int(float(input_dataset['precision'][0]))  # user defined
        precision = int(float(input_dataset['precision'][0]))
        logger.info("!! Numerical precision for calculations is set to {} digits".format(mp.mp.dps))
    else:
        mp.mp.dps = 2 * mp.mp.dps
        precision = 15

    # Defining number of digits for printing Lij outputs
    if input_dataset.get('ndigits') is not None:
        n_digits = int(float(input_dataset['ndigits'][0]))
    else:
        n_digits = 6  # default n.digits
    if n_digits <= precision: # Check that n_digits <= precision
        logger.info("Will print results with {:d} decimal digits.".format(n_digits))
    else:
        n_digits = precision
        logger.info("WARNING! Required number of decimal digits ({:d}) is larger than numerical precision ({:d}). Setting ndigits to {:d}.".format(n_digits, precision, precision))

    # Prepare strings for correct n.digits
    lij_digit_tag = '{{:^+{:d}.{:d}E}}'.format(n_digits + 11, n_digits)
    lij_title_digit_tag = '{{:^{:d}s}}'.format(n_digits + 11)


    # defining T symbolic value (temperatures)
    T = sym.Symbol('T')
    e = sym.Symbol('e')
    kB = mp.mpf("1.0") / mp.mpf("11604.5221")  # Boltzmann constant
    tau = sym.Symbol('tau')  # Indicator of exit/entry jump: 0=exit, 1=entry (for ballistic)
    
    # setting up the random error parameter
    rdshift = {}
    redif = 0.0  # range for random number generation
    remin = 0.0  # minimum for random number generation
    for k in range(len(uniqueconf)):
        rdshift['c'+str(k)] = False  # True means this configuration will be shifted
    for k in range(len(uniquefreq)):
        rdshift['j' + str(k)] = False  # True means this jump frequency will be shifted
    if input_dataset.get('randerror') is not None:
        if len(input_dataset['randerror'])==1 or \
            (len(input_dataset['randerror'])>2 and \
            (input_dataset['randerror'][1]=="allbut" or input_dataset['randerror'][1]=="nonebut")):
            redif=2*np.abs(np.float(input_dataset['randerror'][0]))
            remin=-np.abs(np.float(input_dataset['randerror'][0]))
            del input_dataset['randerror'][0]
        elif len(input_dataset['randerror'])==2 or \
            (len(input_dataset['randerror']) > 2 and \
            (input_dataset['randerror'][2] == "allbut" or input_dataset['randerror'][2] == "nonebut")):
            redif = np.float(input_dataset['randerror'][1])-np.float(input_dataset['randerror'][0])
            remin = np.float(input_dataset['randerror'][0])
            del input_dataset['randerror'][0:2]
        else:
            produce_error_and_quit("Problem in the number/order of the arguments of the RANDERROR keyword.")
        if len(input_dataset['randerror'])>0:
            if input_dataset['randerror'][0] == "allbut":
                for k in rdshift:
                    rdshift[k] = True
            for k in input_dataset['randerror'][1:]:
                if "-" in k:
                    for l in range(int(k[1:].split(sep="-")[0]), int(k[1:].split(sep="-")[1])+1):
                        input_dataset['randerror'].append(k[0]+str(l))
            for k in input_dataset['randerror'][1:]:
                if "-" not in k:
                    if input_dataset['randerror'][0] == "allbut":
                        rdshift[k] = False
                    elif input_dataset['randerror'][0] == "nonebut":
                        rdshift[k] = True
        else:
            for k in rdshift:
                rdshift[k] = True

    # defining symbolic configurations and jump frequencies
    C = sym.MatrixSymbol('C', len(uniqueconf), 1)
    # ajuster la taille de dEs lorsque le rayon thermo=rayon cinétique
    S = sym.MatrixSymbol('S', len(uniqueconf), 1) # energy differences between stationary and equilibrium states
    Snum = mp.mpf(1.) * np.ones((S.shape[0], 1), dtype=float)  # initializing energy differences
    W = sym.MatrixSymbol('W', len(uniquefreq), 1)

    # Set up list of components for each species (spec_list)
    spec_list = [[0,0] for _ in range(Klambda.shape[1])]
    for c in complist:
        spec_list[c.get_species().get_index()-1][0] = c.get_species().get_name()
        spec_list[c.get_species().get_index()-1][1] += 1
    logger.info("The cluster contains:")
    for item in spec_list:
        if item[1] != 0:
            logger.info("  {} {}".format(item[1], item[0]))

    # Output file name
    if input_dataset.get('batch') is not None:
        if input_dataset.get('output') is not None:
            outputfile = dir + 'batch/' + input_dataset['output'][0]
        else:
            outputfile = dir + 'batch/' + 'numerical_kineclue_output'
    else:
        if input_dataset.get('output') is not None:
            outputfile = dir + input_dataset['output'][0]
        else:
            outputfile = dir + 'numerical_kineclue_output'

    # Reading crystal (unstrained crystal)
    if os.path.exists(dir + 'crystal_' + crystal_name + '.pkl'):
        with open(dir + 'crystal_' + crystal_name + '.pkl', 'rb') as input:
            crystal, symop = pickle.load(input)
    else:
        produce_error_and_quit("Crystal file not found in directory {}".format(dir))

    # Reading strained crystal (if it exists)
    if os.path.exists(dir + 'crystal_' + crystal_name + '_strained.pkl'):
        logger.info("Strain crystal found!")
        strain_flag = True
        if input_dataset.get('numstrain') is None:
            logger.info("!! WARNING, found strained crystal but no numerical strain range was found in input file. Continuing with strained crystal but eps = 0.")
        with open(dir + 'crystal_' + crystal_name + '_strained.pkl', 'rb') as input:
            with sym.evaluate(False):
                strcrystal, _ = pickle.load(input)
    else:
        if input_dataset.get('numstrain') is not None:
            logger.info("!! WARNING, found numerical strain range, but no strained crystal in directory {}. Continuing with calculation without strain.".format(dir))
        strain_flag = False
        strcrystal = crystal

    # Analyzing configurations and jumps in the interaction model file
    interconf = []  # initializing list of interactions for configurations
    interjump = []  # initializing list of interactions for jump frequencies
    intersubjump = []  # initializing list of interactions for subjump frequencies
    if input_dataset.get('interactionmodel') is not None:
        logger.info("Reading and identifying configurations and jump frequencies in file {}".format(input_dataset['interactionmodel'][0]))
        # Check if interaction file exists
        if os.path.exists(input_dataset['interactionmodel'][0]):
            # Copy interaction file into result directory
            if not os.path.exists(dir+input_dataset['interactionmodel'][0].split("/")[-1]):  # if interaction file is not already in output directory
                copyfile(input_dataset['interactionmodel'][0], dir+input_dataset['interactionmodel'][0].split("/")[-1])
            else:
                logger.info("!! Will not copy the interaction model file because one already exists in the folder")
        else:
            produce_error_and_quit("Interaction file {} not found.".format(input_dataset['interactionmodel'][0]))
        # Creating list of species from analytical output
        speclistdict = {'bulk': bulkspec}
        for s in set([s.get_species() for s in complist]):
            speclistdict[s.get_name()] = s
        input_list = [s.splitlines() for s in open(input_dataset['interactionmodel'][0], mode="r", encoding='utf-8').read().lower().replace('#&','&#').split(sep='&')[1:] if s[0]!='#']
        input_list = [[y.split(sep="#")[0] for y in x] for x in input_list if x != '']
        for idx, inter in enumerate(input_list): # loop on all interactions in the file
            inter = [s.split() for s in inter]
            spok = True
            ncons = int(float(inter[0][2]))
            interspec = [sp[1] for sp in spec_list]  # for comparison with cluster
            for k in range(0, ncons): # loop on "constraints" or "components"
                # check that species exists
                if speclistdict.get(inter[k+1][0]) is None:
                    logger.info('!! In {} file, species {} of interaction {} does not appear in the analytical input: this interaction is ignored'.format(input_dataset['interactionmodel'][0], inter[k+1][0], idx + 1))
                    spok = False
                    break
                else:
                    if speclistdict[inter[k+1][0]].get_index() != 0:
                        interspec[speclistdict[inter[k+1][0]].get_index()-1]-=1
            if any([a < 0 for a in interspec]) or (any([a != 0 for a in interspec]) and inter[0][0]=='j'): #second condition is added to remove subjumps
#            if any([a < 0 for a in interspec]): #second condition was removed to account for subjumps
                logger.info('!! In {} file, the number of components per species does not match that of the current cluster: this interaction is ignored'.format(input_dataset['interactionmodel'][0]))
                spok = False
            if spok:
                if inter[0][0] == 'c': # configuration
                    specs = []
                    allvec = []
                    for k in range(0, ncons): # loop on components or constraints
                        vec = np.array([evalinput(s) for s in inter[k+1][1:4]], dtype=float)
                        if inter[0][1] == 'o':
                            vec = change_base(arr=vec, crystal=crystal)
                        if k == 0:
                            t = -np.floor(vec + tol_pbc)
                        allvec.append(vec+t)
                        specs.append(speclistdict[inter[k+1][0]])
                    # Creating permutation list
                    tmp = []
                    for sp in speclistdict:
                        indices = []
                        for i, x in enumerate(specs):
                            if x == speclistdict[sp]:
                                indices.append(i)
                        tmp.append(list(permutations(indices)))
                    perms = list(product(*tmp))
                    for i, x in enumerate(perms):
                        perms[i] = flat_list(x)
                    symallvec, symidx = apply_symmetry_operations(vector_list=allvec, symop_list=symop, symop_indices=True)
                    symallvec, symidx = trans_and_check_unicity(veclist=symallvec, symop_idx_list=symidx, perms=perms, species_list=specs, dbdist=dbdist,
                                                                all_defect_list=all_defect_list, bulkspecies_list=bulkspecies_list, bulkspecies_symdefects=bulkspecies_symdefects)
                    specs = [specs[i] for i in perms[0]]  # reordering species according to the order they will come out of trans_and_check_unicity
                    for i, symvec in enumerate(symallvec):
                        deflist = []
                        translist = []
                        for k in range(0, ncons):
                            [defect, trans] = vect2deftrans(vec=np.array(symvec[k], dtype=float), all_defect_list=all_defect_list)
                            deflist.append(defect)
                            translist.append(trans)
                        if len(inter[0]) == 5 or len(inter[0]) == 14:  # entropy factor has been given
                            Eb = mp.mpf(inter[0][4])
                            Sb = mp.mpf(inter[0][3])
                        elif len(inter[0]) == 4 or len(inter[0]) == 13:  # entropy factor was not given
                            Eb = mp.mpf(inter[0][3])
                            Sb = mp.mpf("1.0")
                        else:
                            logger.info("!! Invalid number of data to characterize the configuration with binding energy or prefactor {}".format(inter[0][3]))
                        if len(inter[0]) == 13 or len(inter[0]) == 14:
                            elasticdipole = np.array([float(e) for e in inter[0][-9:]]).reshape(3,3)
                            if inter[0][1] == 'o':
                                elasticdipole = change_base(arr=elasticdipole, crystal=crystal, matrix=True)
                            # Symmetry operation on stress tensor
                            elasticdipole = rotate_matrix(matrix=elasticdipole, symop=symop[symidx[i]])
                            # Convert to coordinates in orthonormal base
                            elasticdipole = change_base(arr=elasticdipole, crystal=crystal, inv=True, matrix=True)
                        else:
                            elasticdipole = np.zeros((3, 3), dtype='float')
                        interconf.append(Subconfiguration(Ebind=Eb+np.sum(np.multiply(elasticdipole, strcrystal.get_orthostrain())), Sbind=Sb,
                                                          species=copy.copy(specs), defects=copy.copy(deflist),
                                                          translations=copy.copy(translist), complist=complist))
                        if i == 0: # checking connectivity (on unstrained crystal)
                            if check_connectivity(configuration=interconf[-1], crystal=crystal, kira=ThRa)[0]:
                                logger.info("  Found {:3.0f} symmetry equivalents for sub-configuration with binding energy {} eV and prefactor {}".format(len(symallvec), Eb, Sb))
                            else:
                                del interconf[-1] # removing this interaction from the list
                                logger.info("!!  Sub-configuration with binding energy {} eV and prefactor {} is ignored because it is not connected!".format(Eb, Sb))
                                break

                elif inter[0][0] == 'j': # jump frequency
                    specs = []
                    allvec = []
                    ignore = False
                    for inifin in [0, 1]: #all initial position vectors, then all final position vectors
                        for k in range(0, ncons):  # loop on components or constraints
                            if inter[k + 1][4] != '>':
                                logger.info("!! Jump constraint is not well defined for sub-jump with saddle-point energy or prefactor {}. It is ignored.".format(inter[0][3]))
                                ignore = True
                                break
                            vec = np.array([evalinput(s) for s in inter[k + 1][1+4*inifin:4+4*inifin]], dtype=float)
                            if inter[0][1] == 'o':
                                vec = change_base(arr=vec, crystal=crystal)
                            if k == 0 and inifin == 0:
                                t = -np.floor(vec + tol_pbc)
                            allvec.append(vec + t)
                            if inifin == 0:
                                specs.append(speclistdict[inter[k + 1][0]])
                        if ignore:
                            break
                    srtdsub=sorted([e.get_name() for e in specs if e.get_name() != 'bulk'])
                    if (len(srtdsub)==1):
                        logger.info("!! Sub-jump with saddle-point energy or prefactor {} is ignored because it characterizes a monomer. Monomer jumps should be dealt with the KRAACTIVATION keyword parameters.".format(inter[0][3]))
                    elif not ignore:
                        subj = False  # sub-jump tag
                        if len(srtdsub) < len(sorted([e.get_species().get_name() for e in complist if e.get_species().get_name() != 'bulk'])):
                            subj=True
                        # Creating permutation list
                        tmp = []
                        for sp in speclistdict:
                            indices = []
                            for i, x in enumerate(specs):
                                if x == speclistdict[sp]:
                                    indices.append(i)
                            tmp.append(list(permutations(indices)))
                        perms = list(product(*tmp))
                        for i, x in enumerate(perms):
                            perms[i] = flat_list(x)*2
                        specs2 = list(specs+specs)
                        specs = [specs[i] for i in perms[0]]  # reordering species according to the order they will come out of trans_and_check_unicity
                        for i, x in enumerate(perms):
                            for ii, s in enumerate(x[int(len(specs)/2):]):
                                perms[i][ii+int(len(specs)/2)] += int(len(specs)/2) # ensuring coherent ordering between final and initial positions
                        symallvec, symidx = apply_symmetry_operations(vector_list=allvec, symop_list=symop, symop_indices=True)
                        #Check if reverse jump is included in symmetry equivalent
                        allvec =  allvec[ncons:] + allvec[:ncons] #inverting final and initial constraints; for now initial and final species are kept identical for each constraint
                        found = False
                        for symec in symallvec:
                            if are_equal_arrays(np.array(symec), np.array(allvec)):
                                found = True
                                break
                        if not found: # reverse jump must be added
                            tmp_symallvec, tmp_symidx = apply_symmetry_operations(vector_list=allvec, symop_list=symop, symop_indices=True)
                            symallvec += tmp_symallvec
                            symidx += tmp_symidx
                        symallvec, symidx = trans_and_check_unicity(veclist=symallvec, symop_idx_list=symidx, perms=perms, species_list=specs2, dbdist=dbdist,
                                                                    all_defect_list=all_defect_list, bulkspecies_list=bulkspecies_list, bulkspecies_symdefects=bulkspecies_symdefects)
                        for i, symvec in enumerate(symallvec):
                            deflist = []
                            translist = []
                            for k in range(0, ncons):
                                [defect, trans] = vect2deftrans(vec=np.array(symvec[k], dtype=float), all_defect_list=all_defect_list)
                                deflist.append(defect)
                                translist.append(trans)
                            if len(inter[0]) == 5 or len(inter[0]) == 14:  # entropy factor has been given
                                Eb = mp.mpf(inter[0][4])
                                Sb = mp.mpf(inter[0][3])
                            elif len(inter[0]) == 4 or len(inter[0]) == 13:  # entropy factor was not given
                                Eb = mp.mpf(inter[0][3])
                                Sb = mp.mpf("1.0")
                            else:
                                logger.info("!! Invalid number of data to characterize the jump with saddle-point energy or prefactor {}".format(inter[0][3]))
                            if len(inter[0]) == 13 or len(inter[0]) == 14:
                                elasticdipole = np.array([float(e) for e in inter[0][-9:]]).reshape(3,3)
                                if inter[0][1] == 'o':
                                    elasticdipole = change_base(arr=elasticdipole, crystal=crystal, matrix=True)
                                # Symmetry operation on stress tensor
                                elasticdipole = rotate_matrix(matrix=elasticdipole, symop=symop[symidx[i]])
                                # Convert to coordinates in orthonormal base
                                elasticdipole = change_base(arr=elasticdipole, crystal=crystal, inv=True, matrix=True)
                            else:
                                elasticdipole = np.zeros((3, 3), dtype='float')
                            interjump.append([Subconfiguration(Ebind=Eb-np.sum(np.multiply(elasticdipole, strcrystal.get_orthostrain())), Sbind=Sb,
                                                          species=copy.copy(specs[:ncons]), defects=copy.copy(deflist),
                                                          translations=copy.copy(translist), complist=complist)])
                            deflist = []
                            translist = []
                            if i == 0: # checking connectivity
                                if not check_connectivity(configuration=Configuration(defects=interjump[-1][0].get_nbdefects(), translations=flat_list(interjump[-1][0].get_nbtranslations())), crystal=crystal, kira=ThRa)[0]:
                                    del interjump[-1] # removing this interaction from the list
                                    logger.info("!! Sub-jump with saddle-point energy {} eV and prefactor {} is ignored because it is not connected!".format(Eb, Sb))
                                    break
                            for k in range(0, ncons):
                                [defect, trans] = vect2deftrans(vec=np.array(symvec[k+ncons], dtype=float), all_defect_list=all_defect_list)
                                deflist.append(defect)
                                translist.append(trans)
                            interjump[-1].append(Subconfiguration(Ebind=Eb-np.sum(np.multiply(elasticdipole, strcrystal.get_orthostrain())), Sbind=Sb,
                                                     species=copy.copy(specs[ncons:]), defects=copy.copy(deflist),
                                                     translations=copy.copy(translist), complist=complist))
                            # Computing the net displacement vectors per species
                            netdisp = np.zeros((len(spec_list)+1, 3))
                            for k in range(ncons):
                                netdisp[specs[k].get_index(), :] += symvec[ncons + k, :] - symvec[k, :]
                            interjump[-1].append(netdisp[1:, :])
                            if i == 0: # checking connectivity
                                if not check_connectivity(configuration=Configuration(defects=interjump[-1][1].get_nbdefects(), translations=flat_list(interjump[-1][1].get_nbtranslations())), crystal=crystal, kira=ThRa)[0]:
                                    del interjump[-1] # removing this interaction from the list
                                    logger.info("!! Sub-jump with saddle-point energy {} eV and prefactor {} is ignored because it is not connected!".format(Eb, Sb))
                                    break
                                else:
                                    logger.info("  Found {:3.0f} symmetry equivalents for jump with saddle-point energy {} eV and prefactor {}".format(len(symallvec), Eb, Sb))
                            if subj:
                                intersubjump.append(interjump[-1])
                                del interjump[-1]
                else:
                    logger.info('!! In {} file, interaction {} is neither a configuration C nor a jump frequency J: it is ignored'.format(input_dataset['interactionmodel'][0], idx+1))
        del symop

    # Temperature range for numerical output
    Tmin = float(input_dataset['temperatures'][0])
    Tmax = float(input_dataset['temperatures'][1])
    Tstep = float(input_dataset['temperatures'][2])
    temperature_list = np.arange(Tmin, Tmax + 0.1*Tstep, Tstep, dtype=float)

    # Strain range for numerical output
    if input_dataset.get('numstrain') is not None  and  strain_flag:
        Smax = float(input_dataset['numstrain'][1])
        Sstep = float(input_dataset['numstrain'][2])
        strain_list = np.arange(float(input_dataset['numstrain'][0]), Smax + 0.1*Sstep, Sstep, dtype=float)
    else:
        Smax = 0
        strain_list = np.array([0])

    if input_dataset.get('freqcheck') is None:
        logger.info("!! Will not check that migration barriers are positive (FREQCHECK keyword not found in the numerical input).")
        if len(strain_list) > 1 or strain_list[0] != 0.0:
            logger.info("!! Will not write the strain_num_conf and strain_num_freq files")
    else:
        if len(input_dataset['freqcheck']) == 0:
            a = 3  # number of values to check in the loop (default value is 3)
        elif len(input_dataset['freqcheck']) == 1:
            a = int(input_dataset['freqcheck'][0])  # number of values to check in the loop
        checkstrainlist = [strain_list[a] for a in np.linspace(start=0, stop=len(strain_list)-1, num=min(a, len(strain_list)),endpoint=True, dtype=int).tolist()]
        checktemplist = [temperature_list[a] for a in np.linspace(start=0, stop=len(temperature_list)-1, num=min(a, len(temperature_list)), endpoint=True, dtype=int).tolist()]

    # Kinetic range loop for convergence study
    if input_dataset.get('kiraloop') is not None:
        if len(input_dataset['kiraloop']) == 0:
            input_dataset['kiraloop'].append(1)
        else:
            input_dataset['kiraloop'][0] = int(input_dataset['kiraloop'][0])
        kiraloop = sorted([list(maxkira.keys())[k] for k in range(0, len(maxkira.keys()), input_dataset['kiraloop'][0])], reverse=True)
        if len(kiraloop) == 0:
            logger.info("!! Will not perform automatic convergence study (KIRALOOP keyword was not specified in the analytical input)")
        else:
            if not os.path.isdir(dir + 'CONVERGENCE_STUDY'):
                os.makedirs(dir + 'CONVERGENCE_STUDY')
    else:
        kiraloop = []
        logger.info("!! Will not perform automatic convergence study (KIRALOOP keyword not found in the numerical input)")

    # Calculation mode
    mode_dict = {'Lij': True, 'mob': False}
    if input_dataset.get('mob') is not None:
        mode_dict['mob'] = True

    # Output options
    outopt = {'ND': False, 'UC': False, 'EX': False, 'DR': False, 'CF': False}
    ndir = 1
    if input_dataset.get('outoptions') is not None:
        for m in input_dataset['outoptions']:
            if outopt.get(m) is not None:
                outopt[m] = True
                if m == 'UC' or m == 'DR' or m == 'CF':
                    logger.info('!! Will output {} files'.format(m))

    if not mode_dict['mob']:  # cannot compute exchange if mobility and Lij are not both computed
        if outopt['EX']:
            logger.info('!! Will not output EX files (MOB keyword not found)')
        outopt['EX'] = False
    elif outopt['EX']:
        logger.info('!! Will output EX files')

    if outopt['ND']:
        if L0mat.shape[0] == 1:
            ndir = 1  # number of directions to be printed
            outopt['ND'] = False
            logger.info('!! Will not output ND files (NORMAL keyword was not specified in the analytical input)')
        else:
            ndir = 3
            logger.info('!! Will output ND files')

    if input_dataset.get('sensitivity') is not None and input_dataset.get('precision') is not None:
        input_dataset['sensitivity'] = None
        logger.info('!! Will not be perform sensitivity study because the PRECISION keyword is used')

    if ndir == 1 and input_dataset.get('sensitivity') is not None:
            input_dataset['sensitivity'][0] = '1'  # enforcing sensitivity study on main direction only if other directions are not computed

    # Setting configuration and jump frequency files for batch calculation
    if input_dataset['batchcreate'] is not None:
        batchcreatefunc(ipt=input_dataset['batchcreate'], wdir=dir)
    conf_files = [dir + 'configurations.txt']
    freq_files = [dir + 'jump_frequencies.txt']
    if input_dataset['batch'] is not None:
        if not os.path.exists(dir + 'batch'):
            os.mkdir(dir + 'batch')
        outputfile0 = outputfile
        input_dataset['batch'][1] = str(1+int(input_dataset['batch'][1]))
        for i in range(*[int(e) for e in input_dataset['batch'][0:3]]):
            if input_dataset.get('random') or input_dataset.get('randerror') is not None: #keep the same names
                conf_files.append(dir + 'configurations.txt')
                freq_files.append(dir + 'jump_frequencies.txt')
            else: #user must create freqs and conf files for batch calculations in a subdirectory called 'batch'
                conf_files.append(dir + 'batch/' + 'configurations_' + str(i) + '.txt')
                freq_files.append(dir + 'batch/' + 'jump_frequencies_' + str(i) + '.txt')
        del conf_files[0], freq_files[0]

    # Lambdifying matrices
    logger.info("Lambdify matrices (starts at {:.3f} s)".format(tm.time() - start_time))
    Zfunc = sym.lambdify((C,), zfunc, "mpmath", dummify=False)
    L0func = sym.lambdify((e, W), Array(L0mat), "mpmath", dummify=False)
    Tdfunc = sym.lambdify((W,), Array(Tdiag), "mpmath", dummify=False)
    KLfunc = sym.lambdify((e, W), Array(Klambda), "mpmath", dummify=False)
    if len(transmat) > 0:
        transfunc = sym.lambdify((W,), Array(transmat), "mpmath", dummify=False)
    else:
        transfunc = sym.lambdify((W,), [], "mpmath", dummify=False)
    Dfunc = {} # empty dictionary
    for a in dissocoeff:
        Dfunc[a] = sym.lambdify((W,), dissocoeff[a], "mpmath", dummify=False)
    if len(Tsparse) > 0:
        Tcfunc = sym.lambdify((W,), Array(Tsparse), "mpmath", dummify=False)
    else:
        Tcfunc = sym.lambdify((W,), [], "mpmath", dummify=False)
    for k1 in maxkira:
        for k2 in maxkira[k1][2]:
            for k3 in maxkira[k1][2][k2]:
                maxkira[k1][2][k2][k3] = sym.lambdify((e,), Array(maxkira[k1][2][k2][k3]), "mpmath", dummify=False)
    if len(SiteCorr) > 0:
        SiteCorrFunc = [sym.lambdify((e, W), Array(x), "mpmath", dummify=False) for x in SiteCorr] #dmap, dlambda, dissocontrib
    else:
        SiteCorrFunc = []

    with sym.evaluate(False):
        maxkira2 = copy.deepcopy(maxkira) # copying maxkira in case we need to remove equations

    for fileN, [conf_file, freq_file] in enumerate(zip(conf_files, freq_files)):
        if input_dataset['batch'] is not None:
            logger.info('-----------BATCH CALC {}-----------'.format(str(range(*[int(e) for e in input_dataset['batch'][0:3]])[fileN])))
            outputfile = outputfile0 + '_'+str(range(*[int(e) for e in input_dataset['batch'][0:3]])[fileN])
        # Reading configuration file reduced by thermodynamic radius
        # binding energy convention: positive means attraction
        logger.info("Reading configuration file (starts at {:.3f} s)".format(tm.time()-start_time))
        config_values = {} # initializing dictionary of configuration energy values
        input_list = open(conf_file).read().splitlines()
        input_list = [x for x in input_list if x != '']
        tofile = []
        tofile_strain = []
        eqtoremove = []
        #logger.info("\n\n!!!!! I AM USING SOMETHING VERY SPECIFIC FOR B4C !!!!!\n\n")
        #eqtoremove = [1]  # Special case for B4C, remove afterwards
        for id, myconf in enumerate(input_list[1:]):  # loop over all configurations in the file
            tmp = myconf.split()
            if rdshift["c" + tmp[0]]:
                rderr = mp.mpf(str(-(redif * rd.random() + remin))) # minus sign is because of the binding energy convention
            else:
                rderr = mp.mpf("0")
            firstfound = True
            if int(float(tmp[1])) == -1:  # we use the default value
                if len(interconf) > 0:  # using interaction model
                    conf = uniqueconf[int(tmp[0])]
                    Eb = mp.mpf("0")
                    Sb = mp.mpf("1")
                    for inter in interconf:
                        [ei, si, found] = inter.isincluded(conf=conf)
                        if found:
                            if fileN == 0:
                                if firstfound:
                                    firstfound = False
                                    logger.info("  Looking for sub-configurations in configuration {}".format(conf.get_thermoint()))
                                logger.info("    Found sub-configuration with binding energy {} eV and prefactor {}".format(ei, si))
                            Eb += ei
                            Sb *= si
                    Eb += rderr
                    config_values[str(tmp[0])] = kB*T*sym.log(Sb*sym.exp(Eb/(kB*T)))
                    tofile.append([tmp[0], float(Sb), float(sym.simplify(Eb).subs(e, 0.0)), tmp[3], tmp[4]])
                    if input_dataset.get('freqcheck') is not None:
                        for strain in checkstrainlist:
                            tofile_strain.append([tmp[0], float(Sb), float(sym.simplify(Eb).subs(e, strain)), tmp[3], tmp[4], strain*100, " "])
                    if abs(Sb) < tol:
                        eqtoremove += uniqueconf[int(tmp[0])].get_kineticinter()
                elif input_dataset.get('random') is not None: # random interaction
                    if str(tmp[0]) == "0":  # first dissociated configurations set to zero interaction in random mode
                        config_values[str(tmp[0])] = mp.mpf("0")
                        tofile.append([tmp[0], 1, 0, tmp[3], tmp[4]])
                    else:
                        config_values[str(tmp[0])] = mp.mpf((rd.random()*(float(input_dataset['random'][1])-float(input_dataset['random'][0]))+float(input_dataset['random'][0])))
                        tofile.append([tmp[0], 1, float(config_values[str(tmp[0])]), tmp[3], tmp[4]])
                else: #(non interacting configuration)
                    config_values[str(tmp[0])] = rderr
                    tofile.append([tmp[0], 1, float(rderr), tmp[3], tmp[4]])
            elif float(tmp[1]) < 0:
                logger.info("!! Prefactor is negative. The only negative value accepted is the default one -1; Configuration {} is ignored.".format(tmp[0]))
                config_values[str(tmp[0])] = sym.log(0)
                tofile.append([tmp[0], 0, 0, tmp[3], tmp[4]])
                eqtoremove += uniqueconf[int(tmp[0])].get_kineticinter()
            else:  # using user inputs; if tmp[1] = 0 the configuration will be ignored (user choice)
                config_values[str(tmp[0])] = kB*T*sym.log(mp.mpf(tmp[1]) * sym.exp((rderr+mp.mpf(tmp[2]))/(kB*T)))
                tofile.append([tmp[0], float(tmp[1]), float(rderr)+float(tmp[2]), tmp[3], tmp[4]])
                # Kinetic equations to be removed because associated with configurations with zero prefactor
                if abs(float(tmp[1])) < tol**2:
                    eqtoremove += uniqueconf[int(tmp[0])].get_kineticinter()
            if tofile_strain != []:
                tofile_strain[-1][-1] = "\n"
        if id+1 != C.shape[0]:
            logger.info("!! WARNING! {} configurations are missing from the <configurations.txt> file. Missing configurations are ignored from the calculation and are likely to cause an error. You should not remove the configuration from the file, but rather set its prefactor to 0.".format(C.shape[0]-id-1))
        if input_dataset.get('batch') is None:
            numconffile = dir + 'num_conf.txt'
            numconffile_strain = dir + 'strain_num_conf.txt'
        else:
            numconffile = dir + 'batch/' + 'num_conf_' + str(range(*[int(e) for e in input_dataset['batch'][0:3]])[fileN]) + '.txt'
            numconffile_strain = dir + 'batch/' + 'strain_num_conf_' + str(range(*[int(e) for e in input_dataset['batch'][0:3]])[fileN]) + '.txt'
        with open(numconffile, 'w') as output:
            output.writelines(input_list[0]+"\n")
            for item in tofile:
                output.writelines(("{:5s} {:22.10e} {:18.10f} "+disfmt+" {:8s}\n").format(*item))
        if len(strain_list) > 1 or strain_list[0] != 0.0:
            if tofile_strain != []:
                with open(numconffile_strain, 'w') as output:
                    output.writelines(input_list[0].split(sep="6)")[0]+"6) strain (%)\n\n")
                    for item in tofile_strain:
                        output.writelines(("{:5s}  {:22.10e} {:18.10f} "+disfmt+" {:8s} {:10.6f} {:s}\n").format(*item))
        del myconf, tmp, input_list, id, conf_file, numconffile, numconffile_strain, tofile, tofile_strain

        #  Reading jump frequency file reduced by thermodynamic radius
        logger.info("Reading jump frequency values from the file (starts at {:.3f} s)".format(tm.time() - start_time))
        # Reading in activation energies and jump prefactor for KRA
        kra_sp = {} # initializing dictionary that will read these values
        ballisticjumps = {} # dictionary containing the names of ballistic mechanisms, those with negative activation energy
        input_list = input_dataset['kraactivation']
        tmpkrasp = {}
        # input should have saddle point energies with respect to a cluster configuration
        # where all species are isolated from each other (positive or negative value)
        average_pref = mp.mpf("1")
        average_Esp = mp.mpf("0")
        nk = 0
        for item in input_list:
            if len(item) == 2 or len(item) == 12:  # only energy or energy+elastic dipole
                kra_sp[item[0]] = sym.exp(-mp.mpf(item[1])/(kB*T))
                average_Esp += mp.mpf(item[1])
                nk += 1
            elif len(item) == 3 or len(item) == 13:  # a different prefactor was given
                if float(item[1])==0:
                    kra_sp[item[0]] = mp.mpf("0")
                else:
                    average_pref *= mp.mpf(item[1])
                    if float(item[2]) < -tol: #ballistic jumps are identified with a negative KRA activation energy
                        ballisticjumps[item[0]] = 1
                        kra_sp[item[0]] = mp.mpf(item[1])
                    else:
                        average_Esp += mp.mpf(item[2])
                        kra_sp[item[0]] = mp.mpf(item[1]) * sym.exp(-mp.mpf(item[2]) / (kB * T))
                    nk += 1
            else:
                produce_error_and_quit("!! Incorrect number of inputs for KRA parameters in jump{}".format(item[0]))
            if len(item) == 12 or len(item) == 13: # an elastic dipole was given for isolated mechanisms
                elasticdipole = np.array([float(a) for a in item[-9:]]).reshape(3,3)
                if item[-10] == 's': # we must convert to orthonormal
                    elasticdipole = change_base(arr=elasticdipole, crystal=crystal, inv=True, matrix=True)
                elif item[-10] != 'o':
                    produce_error_and_quit("Bad description of dipole for KRA. Probably missing o or s letter")
            else:
                elasticdipole = np.zeros((3,3))
            tmpkrasp[item[0]] = [kra_sp[item[0]], change_base(arr=elasticdipole, crystal=crystal, inv=False, matrix=True)]
            kra_sp[item[0]]*=sym.exp(np.sum(np.multiply(elasticdipole, strcrystal.get_orthostrain()))/ (kB * T))
        for item in brokenjumps:
            if kra_sp.get(item[0]) is None: # if a broken symmetry jump created under strained was not assigned a KRA value, it is deuced from symmetry operations
                sourcejp = "_".join(item[0].split(sep=("_"))[:-1])
                if ballisticjumps.get(sourcejp) is not None:
                    ballisticjumps[item[0]] = 1
                # Symmetry operation on stress tensor and convert to coordinates in orthonormal base
                elasticdipole = change_base(arr=rotate_matrix(matrix=tmpkrasp[sourcejp][1], symop=item[1]), crystal=crystal, inv=True, matrix=True)
                kra_sp[item[0]] = tmpkrasp[sourcejp][0]*sym.exp(np.sum(np.multiply(elasticdipole, strcrystal.get_orthostrain()))/(kB*T))
        # Average jump frequency from jump mechanisms KRA activation energies
        Wmean = (average_pref**(1.0/nk))*sym.exp(-average_Esp/(nk*kB*T))
        # Checking that when the ballistic keyword was not used in the analytical part, the ballisticjumps dict is empty
        if len(transmat)==0 & len(ballisticjumps)!=0:
            produce_error_and_quit("Activation energy was set to a negative value in KRAACTIVATION while the BALLISTIC keyword was not used in the analytical part. This does not make sense.")
        del input_list, average_Esp, average_pref, nk, elasticdipole, item, tmpkrasp

        # Removing kinetic equations
        if len(eqtoremove) > 0:
            maxkira = copy.deepcopy(maxkira2)
            Tdiag2 = Tdiag
            Klambda2 = Klambda
            if len(SiteCorr) > 0:
                Dmap = SiteCorr[0]
            eqtoremove = sorted(eqtoremove, reverse=True)
            tremove = []
            for interaction in eqtoremove:
                Tdiag2 = np.delete(Tdiag2, interaction-1, axis=0)
                Klambda2 = np.delete(Klambda2, interaction-1, axis=2)
                if len(SiteCorr) > 0:
                    Dmap = np.delete(Dmap, interaction-1, axis=0)
                for idx, coords in enumerate(Tcoords):
                    if interaction-1 in coords:
                        tremove.append(idx)
            Tcoords2 = [t for idx, t in enumerate(Tcoords) if idx not in tremove]
            Tsparse2 = [t for idx, t in enumerate(Tsparse) if idx not in tremove]
            # Renumbering Tcoords2 because some equations were removed
            for idx, interaction in enumerate(Tcoords2):
                Tcoords2[idx] = (interaction[0]-len([e for e in eqtoremove if e < interaction[0]+1]), interaction[1]-len([e for e in eqtoremove if e < interaction[1]+1]))
            logger.info("!! Removed {} kinetic equations because some configuration prefactors were set to zero".format(len(eqtoremove)))
            # Re-Lambdifying matrices because some configurations were removed
            logger.info("Re-Lambdify matrices (starts at {:.3f} s)".format(tm.time() - start_time))
            KLfunc = sym.lambdify((e, W), Array(Klambda2), "mpmath", dummify=False)
            Tdfunc = sym.lambdify((W,), Array(Tdiag2), "mpmath", dummify=False)
            if len(SiteCorr) > 0:
                SiteCorrFunc[0] = sym.lambdify((e,W), Array(Dmap), "mpmath", dummify=False)
            if len(Tsparse) > 0:
                Tcfunc = sym.lambdify((W,), Array(Tsparse2), "mpmath", dummify=False)
            else:
                Tcfunc = sym.lambdify((W,), [], "mpmath", dummify=False)
            # Removing kinetic interactions from maxkira as well
            for key in maxkira:
                for k in range(len(maxkira[key][0]) - 1, -1, -1):
                    if maxkira[key][0][k] in eqtoremove:
                        del maxkira[key][0][k]
                    else:
                        maxkira[key][0][k] -= len([a for a in eqtoremove if a < maxkira[key][0][k]])
        else:
            Tcoords2 = Tcoords
        Ninter = int(Klambda.shape[2]-len(eqtoremove))

        # Computing matrix derivatives for sensitivity study if needed
        if input_dataset.get('sensitivity') is not None:  # Computing matrix derivatives
            logger.info("Computing matrices derivatives")
            if not os.path.isdir(dir + 'SENSITIVITY'):
                os.makedirs(dir + 'SENSITIVITY')  # Creating files for sensitivity study output if needed
            derL0 = []
            derKL = []
            derT = []
            derSiteCorrFunc = []
            uniqueJF = []
            avoid = {}
            for i in input_dataset['sensitivity'][1:]:
                if '-' in i:
                    for k in range(int(i.split(sep='-')[0]), int(i.split(sep='-')[1])+1):
                        avoid[str(k)] = 0
                else:
                    avoid[i] = 0
            del input_dataset['sensitivity'][1:]
            for k in range(W.shape[0]):
                if avoid.get(str(k)) is None:  # this jump frequency is fixed by the user and will not be part of sensitivity analysis
                    uniqueJF.append(k)
                    Wnum = np.zeros((W.shape[0], 1), dtype=float)  # initializing jump frequency values
                    Wnum[k][0] = 1.0
                    # Sensitivity study calculations are performed at the maximum strain (0 by default if no strain is specified)
                    dTnum = sparse.csc_matrix((np.array(Tcfunc(Wnum), dtype=float), ([e[0] for e in Tcoords2], [e[1] for e in Tcoords2])), shape=(Ninter, Ninter), dtype=float)
                    dTnum += dTnum.transpose()
                    dTnum += sparse.csc_matrix((np.array(Tdfunc(Wnum), dtype=float)[:, 0], (list(range(Ninter)), list(range(Ninter)))), shape=(Ninter, Ninter), dtype=float)
                    derT.append(dTnum)
                    derL0.append([np.matrix(L0func(Smax, Wnum)[d]) for d in range(ndir)])
                    derKL.append([np.matrix(KLfunc(Smax, Wnum)[d]) for d in range(ndir)])
                    if len(SiteCorrFunc) != 0:
                        derSiteCorrFunc.append([np.matrix(SiteCorrFunc[0](Smax, Wnum)),
                                                [np.matrix(SiteCorrFunc[1](Smax, Wnum)[d]) for d in range(ndir)],
                                                np.matrix(SiteCorrFunc[2](Smax, Wnum))])
            del dTnum

        # Assigning values as a function of strain and temperature for jump frequencies
        input_list = open(freq_file).read().splitlines()
        input_list = [x for x in input_list if x != '']
        tofile = []
        tofile_strain = []
        freq_functions = {}  # dictionary of jump frequencies, containing for each Wx a lambda function Wx(temperature)
        for idxfreq, myfreq in enumerate(input_list[1:]):  # loop over all jump frequencies in the file
            tmp = myfreq.split()
            if rdshift["j" + tmp[0]]:
                rderr = mp.mpf(str(redif * rd.random() + remin))
            else:
                rderr = mp.mpf("0")
            if int(float(tmp[1])) == -1:
                if kra_sp.get(tmp[5]) is None:
                    produce_error_and_quit("Activation energy is not defined for jump {}. Use the same name as in the analytical part of the code".format(tmp[5]))
                else: # we use the default value (KRA)
                    if config_values[tmp[3]] == sym.zoo*T or config_values[tmp[4]] == sym.zoo*T:  # sym.zoo = complex infinity
                        # initial or final configuration is not possible (user choice) so jump is not possible
                        freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau), mp.mpf('0'), "mpmath", dummify=False)
                        tofile.append([tmp[0], 0, 0, tmp[3], tmp[4], tmp[5], tmp[6]])
                    elif kra_sp[tmp[5]] == 0:  # sym.zoo = complex infinity
                        # KRA prefactor was set to 0 so jump is not possible
                        freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau), mp.mpf('0'), "mpmath", dummify=False)
                        tofile.append([tmp[0], 0, 0, tmp[3], tmp[4], tmp[5], tmp[6]])
                    elif ballisticjumps.get(tmp[5]) is not None:
                        freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau),
                                                 kra_sp[tmp[5]]*sym.exp((tau*config_values[tmp[4]].subs(e,0)+(1-tau)*config_values[tmp[3]].subs(e,0))/(kB*T))*
                                                 (tau*S[int(tmp[4]), 0]+(1-tau)*S[int(tmp[3]), 0]) / Wmean, "mpmath", dummify=False)  # Jump frequencies are now functions of strain and temperature
                        tofile.append([tmp[0], float(kra_sp[tmp[5]].subs(e, 0.).subs(T, 0.5*(Tmin+Tmax))), float((-kB*T*sym.log(Wmean*freq_functions[tmp[0]](0,0.5*(Tmin+Tmax),Snum,0)/kra_sp[tmp[5]].subs(e,0).subs(T, 0.5*(Tmin+Tmax)))).subs(T,0.5*(Tmin+Tmax))), tmp[3], tmp[4], tmp[5], tmp[6]])
                    else:
#                        Esp = -kB*T*sym.log(kra_sp[tmp[5]])-mp.mpf("0.5")*(config_values[tmp[3]].subs(e,0)+config_values[tmp[4]].subs(e,0))
                        Esp = -kB*T*sym.log(kra_sp[tmp[5]])-mp.mpf("0.5")*(config_values[tmp[3]]+config_values[tmp[4]])
                        Ssp = mp.mpf("1")
                        if len(interjump+intersubjump) > 0 and ballisticjumps.get(tmp[5]) is None:  # we use the interaction model
                            freq = uniquefreq[int(tmp[0])]
                            if (not (freq.get_beyond() or freq.get_disso())): # for monomer calculations one should use KRA
                                #logger.info("  Looking for sub-jumps in jump {}".format(freq.get_number()))
                                for inter in interjump+intersubjump:   
                                    if np.any([are_equal_arrays(inter[-1], e) for e in jumpmechdisp[freq.get_jump()]]):
                                        [ei, si, foundi] = inter[0].isincluded(conf=freq.get_config_ini())
                                        [ef, sf, foundf] = inter[1].isincluded(conf=freq.get_config_fin())
                                        if foundi and foundf:
                                            if fileN == 0:
                                                logger.info("    In jump {}, found sub-jump with saddle-point energy {} eV and prefactor {}".format(freq.get_number(), ei, si))
                                            Esp = ei
                                            Ssp = si
                                            break #only looks for one jump, the first one
                            elif freq.get_config_ini().get_beyond() and freq.get_config_fin().get_beyond():
                                if len(freq.get_config_ini().get_subconfigurations()) == len(freq.get_config_fin().get_subconfigurations()):
                                    donesub = False
                                    for a,b in zip(freq.get_config_ini().get_subconfigurations(), freq.get_config_fin().get_subconfigurations()):
                                        if not (len(a.get_defects())==len(b.get_defects()) and np.all(a.get_species()==b.get_species())):
                                            donesub=True
                                            # to be analyzed, both configurations should have the same set of subconfigurations before and after the jump
                                    if not donesub:
                                        for inter in intersubjump:
                                            if np.any([are_equal_arrays(inter[-1], e) for e in jumpmechdisp[freq.get_jump()]]):
                                                [ei, si, foundi] = inter[0].isincluded(conf=freq.get_config_ini())
                                                [ef, sf, foundf] = inter[1].isincluded(conf=freq.get_config_fin())
                                                if foundi and foundf:
                                                    if fileN == 0:
                                                        logger.info("    In jump {}, found sub-jump with saddle-point energy {} eV and prefactor {}".format(freq.get_number(),ei, si))
                                                    Esp = ei
                                                    Ssp = si
                                                    break  # only looks for one jump
                        Esp = Esp+rderr
                        freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau), Ssp*sym.exp(-Esp / (kB * T))*(tau*S[int(tmp[4]),0]+(1-tau)*S[int(tmp[3]),0])/Wmean, "mpmath", dummify=False)  # Jump frequencies are now functions of strain and temperature
                        npf = Ssp*(sym.exp(-Esp/(kB*T))).subs(T, mp.mpf(str(10**(2*mp.mp.dps)))).subs(e,0)
                        if npf < tol or Ssp < tol:
                            nesp = 0.
                            npf = 0.
                        else:
                            nesp = float((-kB*T*sym.log(Ssp*sym.exp(-Esp/(kB*T))/npf)).subs(T, 0.5*(Tmin+Tmax)).subs(e,0.0))
                            if input_dataset.get('freqcheck') is not None:
                                check=False
                                for strain in checkstrainlist:
                                    if not check:
                                        for temp in checktemplist:
                                            nesp_strain = float((-kB*T*sym.log(Ssp*sym.exp(-Esp/(kB*T))/npf)).subs(T, temp).subs(e, strain))
                                            # Check if we get negative migration barriers because of strain - in that case, strain is too large and linear elasticity theory does not work any more
                                            try: #if it's a float
                                                tmp_eb_conf = [-float(config_values[tmp[3]]),-float(config_values[tmp[4]])]
                                            except: # if it's symbolic
                                                tmp_eb_conf = [-float(config_values[tmp[3]].subs(T, temp).subs(e, strain)), -float(config_values[tmp[4]].subs(T, temp).subs(e, strain))]
                                            # Print the forward and backward migration energies for check purposes
                                            # print("Freq. #{:3s} ({:3s}) between config #{:3s} and config #{:3s}; migration energies (eV) forward {:.4f} ; backward {:.4f}".format(tmp[0], tmp[5], tmp[3], tmp[4], nesp_strain-tmp_eb_conf[0], nesp_strain-tmp_eb_conf[1]))
                                            # Produce warning if saddle point energy is lower than that of any of the two end configurations
                                            if nesp_strain < tmp_eb_conf[0] or nesp_strain < tmp_eb_conf[1]:
                                                check=True
                                                logger.info("!! WARNING! Negative migration barrier for frequency {:^5s} (T={:4.0f}K S={:+7.4f}%)".format(tmp[0], temp, strain*100))
                                                break
                                    if len(strain_list)>1 or strain_list[0] != 0.0:
                                        # Energy values under strain are computed at mid temperature between Tmin and Tmax
                                        nesp_strain = float((-kB*T*sym.log(Ssp*sym.exp(-Esp/(kB*T))/npf)).subs(T, 0.5*(Tmin + Tmax)).subs(e, strain))
                                        tofile_strain.append([tmp[0], float(npf), float(nesp_strain), tmp[3], tmp[4], tmp[5], tmp[6], strain*100, " "])
                        tofile.append([tmp[0], float(npf), float(nesp), tmp[3], tmp[4], tmp[5], tmp[6]])
            elif int(float(tmp[1])) < 0: # not allowed
                freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau), mp.mpf('0'), "mpmath", dummify=False)
                tofile.append([tmp[0], 0., 0., tmp[3], tmp[4], tmp[5], tmp[6]])
            else:  # using user inputs; if tmp[1] = 0 the jump frequency will be ignored (user choice)
                if ballisticjumps.get(tmp[5]) is None:
                    freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau), (mp.mpf(tmp[1])/Wmean)*(tau*S[int(tmp[4]),0]+(1-tau)*S[int(tmp[3]),0])*sym.exp(-(mp.mpf(tmp[2])+rderr)/(kB*T)),"mpmath", dummify=False)
                    tofile.append([tmp[0], float(tmp[1]), float(tmp[2])+float(rderr), tmp[3], tmp[4], tmp[5], tmp[6]])
                else:
                    freq_functions[tmp[0]] = sym.lambdify((e,T,S,tau), (mp.mpf(tmp[1])/Wmean)*sym.exp((tau*config_values[tmp[4]]+(1-tau)*config_values[tmp[3]])/(kB*T))*(tau*S[int(tmp[4]),0]+(1-tau)*S[int(tmp[3]),0]), "mpmath", dummify=False)
                    tofile.append([tmp[0], float(tmp[1]), float(tmp[2])+float(rderr), tmp[3], tmp[4], tmp[5], tmp[6]])
            if tofile_strain != []:
                tofile_strain[-1][-1] = "\n"
        if idxfreq+1 != W.shape[0]:
            logger.info("!! WARNING! {} jump frequencies are missing from the <jump_frequencies.txt> file. Missing jump frequencies are ignored from the calculation. You should not remove a jump frequency from this file but rather set its prefactor to 0.".format(W.shape[0]-idxfreq-1))
        if input_dataset.get('batch') is None:
            numfreqfile = dir + 'num_freq.txt'
            numfreqfile_strain = dir + 'strain_num_freq.txt'
        else:
            numfreqfile = dir + 'batch/' + 'num_freq_' + str(range(*[int(e) for e in input_dataset['batch'][0:3]])[fileN]) + '.txt'
            numfreqfile_strain = dir + 'batch/' + 'strain_num_freq_' + str(range(*[int(e) for e in input_dataset['batch'][0:3]])[fileN]) + '.txt'
        with open(numfreqfile, 'w') as output:
            output.writelines(input_list[0]+"\n")
            for item in tofile:
                output.writelines(("{:5s} {:22.10e} {:18.10f}   {:5s} {:5s} "+fnfmt+" {:6s}\n").format(*item))
        if len(strain_list) > 1 or strain_list[0] != 0.0:
            if tofile_strain != []:
                with open(numfreqfile_strain, 'w') as output:
                    output.writelines(input_list[0]+"; 8) strain (%)\n\n")
                    for item in tofile_strain:
                        output.writelines(("{:5s} {:22.10e} {:18.10f}   {:5s} {:5s} "+fnfmt+" {:6s} {:10.6f} {:s}\n").format(*item))
        del numfreqfile, numfreqfile_strain, tofile, tofile_strain

        # Lambdifying jump frequencies
        logger.info("Lambdify jump frequencies and configuration probabilities (starts at {:.3f} s)".format(tm.time() - start_time))
        freq_functions_mob = copy.copy(freq_functions)
        for myfreq in uniquefreq:
            if myfreq.get_disso():
                freq_functions_mob[str(myfreq.get_number())] = sym.lambdify((e,T,S,tau), mp.mpf('0'), "mpmath", dummify=False)
        del myfreq, input_list, tmp, freq_file

        # Lambdifying configuration energies
        config_functions = {}
        for myconf in config_values: #lambdifying values; could not do it before because of KRA calculation which requires sym expression and not function
            if config_values[myconf] == sym.zoo*T:
                config_functions[myconf] = sym.lambdify((e,T,S), 0, "mpmath", dummify=False)
            else:
                config_functions[myconf] = sym.lambdify((e,T,S), sym.exp(config_values[myconf]/(kB*T))*S[int(myconf),0], "mpmath", dummify=False)
        del myconf, config_values

        # Setting up the prefactor: a*a*pref*Wmean*units
        JumpPref = Wmean
        if input_dataset.get('lattparam') is not None:
            alat = mp.mpf(input_dataset['lattparam'][0])*mp.mpf("1e-10")
        else:
            alat = mp.mpf("1e-10")
        JumpPref *= alat * alat * mp.mpf("1e12")
        if input_dataset.get('prefactor') is not None:
            JumpPref *= mp.mpf(input_dataset['prefactor'][0])
        # Units dictionary
        volat = strcrystal.get_atomicvolume()*alat*alat*alat
        units_dict = {'m2/s': mp.mpf("1"), 'cm2/s': mp.mpf("1e4"), 'm2/s/eV': mp.mpf("1")/(kB*T), 'cm2/s/eV': mp.mpf("1e4")/(kB*T), '/m/s': mp.mpf("1")/volat, '/m/s/eV': mp.mpf("1")/(volat*(kB*T)), 'debug': mp.mpf("1e8")}
        units = 'm2/s' # default units
        if input_dataset.get('units') is not None:
            if units_dict.get(input_dataset['units'][0]) is not None:
                units = input_dataset['units'][0]
                JumpPref = JumpPref*units_dict[units]
        JumpPrefnum = sym.lambdify((e,T), JumpPref, "mpmath")
        logger.info("Cluster transport coefficients are output in units of {}".format(units))
        del volat

        # Preparing outfiles
        for direction in range(ndir):
            for out in outopt:
                if (out != "ND") and outopt[out]:
                    with open(outputfile + '_' + str(direction) + out + '.dat', 'w') as output:
                        if out == "DR":
                            koutdr = 3
                            output.writelines(("#{:^10s} {:^15s} {:^16s}").format('1) T [K]', '2) Strain [%]', '3) 1000/T [/K]'))
                            for a in range(0, len(spec_list)):
                                for b in range(a+1, len(spec_list)):
                                    koutdr += 1
                                    output.writelines(' {:s}'.format(lij_title_digit_tag).format('{}) L_{}{}/L_{}{}'.format(koutdr, b, a, a, a)))
                                    koutdr += 1
                                    output.writelines(' {:s}'.format(lij_title_digit_tag).format('{}) L_{}{}/L_{}{}'.format(koutdr, a, b, b, b)))
                            output.writelines("\n")
                        else:
                            kout = 3  # printing header
                            output.writelines(("#{:^10s} {:^15s} {:^16s}").format('1) T [K]', '2) Strain [%]', '3) 1000/T [/K]'))
                            for a in range(0, len(spec_list)):
                                for b in range(a, len(spec_list)):
                                    kout += 1
                                    output.writelines(' {:s}'.format(lij_title_digit_tag).format('{}) {}_{}{}'.format(kout, out, a, b)))
                            output.writelines("\n")

        # total number of configurations (partition function with no binding energy)
        Zfunc0 = Zfunc(np.ones((C.shape[0],1), dtype=float))
        logger.info("Total number of configurations  Z0 = {}".format(Zfunc0))

        # CLUSTER ONSAGER MATRIX CALCULATIONS DEPENDING ON MODE

        # If the ballistic keyword was used, one needs a reference configuration to compute the cluster partition function
        if len(transmat) > 0:  # non-equilibrium calculation, we need a reference configuration to compute the partition function
            refdisso = None
            if input_dataset.get('refdisso') is not None and len(input_dataset[refdisso])>0:
                for i in uniqueconf:
                    if i.get_thermoint()==int(input_dataset[refdisso][0]): #configuration wanted by the user
                        refdisso = int(input_dataset[refdisso][0])
                        if i.get_label()[0]!="d":
                            logger.info("!! The reference configuration specified by the user with REFDISSO is not a dissociated configuration.")
                        break
            if refdisso is None:
                logger.info("!! Reference configuration for the computation of the cluster partition function is chosen automatically.")
                nc = 0
                for i in uniqueconf:
                    if i.get_label()[0]=="d" and len(i.get_subname().split(sep="|"))>nc:
                        nc = len(i.get_subname().split(sep="|"))
                        refdisso = i.get_thermoint()
                if refdisso is None: #we have looked for dissociated configurations and did not find any
                    refdisso = uniqueconf[0].get_thermoint()
                    logger.info("!! No dissociated configuration was found to set the reference configuration to compute the cluster partition function.")
            logger.info("!! Configuration # {:.0f} is taken as the reference configuration to compute the cluster partition function.".format(refdisso))
            del i, nc

        logger.info("Computing Onsager matrices (starts at {:.3f} s)".format(tm.time() - start_time))
        tofile = [[] for _ in range(ndir)]
        Llu = None
        if input_dataset.get('precision') is not None:
            Tnum = mp.matrix(Ninter)  # no need to initialize at each step; always the same coefficients are changed
        for m in sorted(mode_dict):
            if not mode_dict[m]:
                continue
            else:
                if m == 'Lij':
                    logger.info("  Computing cluster transport coefficients...")
                    freq_func = freq_functions
                elif m == 'mob':
                    logger.info('--------------------------------------------------------------')
                    logger.info("  Computing cluster mobility...")
                    freq_func = freq_functions_mob
            # Temperature and strain loop
            k=0
            for temp in temperature_list:
                for strain in strain_list:
                    notthatdir = [] # to remove directions where transport coefficients are all zero from sensitivity study
                    if len(strain_list) > 0:
                        logger.info('\nTemperature = {:6.1f} K - Strain = {:+8.4f} % '.format(temp, strain*100))
                    else:
                        logger.info('\nTemperature = {:6.1f} K'.format(temp))
                    Snum = mp.mpf("1")*np.ones((S.shape[0], 1), dtype=float)
                    Wnum = mp.mpf("0")*np.zeros((W.shape[0], 1), dtype=float)
                    Wnum0 = mp.mpf("0")*np.zeros((W.shape[0], 1), dtype=float)
                    for w in freq_func:
                        Wnum[int(w), 0] = freq_func[w](strain, temp, Snum, 1)
                    if len(transmat) > 0:  # non-equilibrium calculation, we need to compute the stationary state
                        if input_dataset.get('precision') is not None:  # arbitrary precision using mpmath
                            Snum = solve_stationary_prec(transmat=transfunc(Wnum), Cref=refdisso)
                        else:
                            Snum = solve_stationary_state(transmat=transfunc(Wnum), Cref=refdisso)
                        for w in freq_func:
                            Wnum[int(w), 0] = freq_func[w](strain, temp, Snum, 1)
                            Wnum0[int(w), 0] = freq_func[w](strain, temp, Snum, 0)
                    else:
                        Wnum0 = Wnum
                    # Computing partition function, dissociation frequency and lifetime
                    Cnum = mp.mpf("0")*np.zeros((C.shape[0],1), dtype=float)
                    for c in config_functions:
                        Cnum[int(c),0] = config_functions[c](strain, temp, Snum)
                    #tofile_Cnum = ' '.join('{:10.3e}'.format(float(e)) for e in Cnum.transpose()[0]) + '\n'
                    #with open(outputfile + '_0P.dat', 'w') as file:
                    #    file.write(tofile_Cnum)
                    zfuncnum = Zfunc(Cnum)
                    alldissonum = {}
                    dissonum = 0 # total dissociation coefficient
                    for a in Dfunc:
                        alldissonum[a] = float((JumpPrefnum(strain, temp) / (alat*alat*zfuncnum)) * Dfunc[a](Wnum))
                        dissonum += alldissonum[a]
                    if m == 'Lij':
                        with warn.catch_warnings():  # avoid printing "divide by zero" warning
                            warn.simplefilter("ignore")
                            logger.info('  Z={:^10.3E}  D={:^10.3E}  t={:^10.3E}\n'.format(float(zfuncnum), dissonum, np.float64(1.)/dissonum))  # np.float64 gives INF if dividing by zero instead of raising exception
                    ################################################
                    #### Solving system of equation numerically ####
                    ################################################
                    if input_dataset.get('precision') is not None: #arbitrary precision using mpmath
                        numval = Tcfunc(Wnum)
                        for x, coords in enumerate(Tcoords2):
                            Tnum[coords] = numval[x]
                            if len(transmat) == 0:
                                Tnum[(coords[1], coords[0])] = numval[x]
                        numval = Tdfunc(Wnum)
                        for x in range(Ninter):
                            Tnum[(x, x)] = numval[x][0]
                        Lij = solve_prec(Tnum=Tnum, Klambda=KLfunc(strain, Wnum), Klambda0=KLfunc(strain, Wnum0), Ninter=Ninter, Nspec=len(spec_list),
                                         Ndir=ndir, Pref=(JumpPrefnum(strain,temp) / zfuncnum), L0=L0func(strain, Wnum),
                                         Site=[x(strain, Wnum) for x in SiteCorrFunc])
                    else: # default precision using scipy
                        Tnum = sparse.csc_matrix((np.array(Tcfunc(Wnum), dtype=np.longdouble), ([a[0] for a in Tcoords2], [a[1] for a in Tcoords2])), shape=(Ninter, Ninter), dtype=np.longdouble)
                        if len(transmat) == 0:
                            Tnum += Tnum.transpose()
                        if not neumann:
                            Tnum += sparse.csc_matrix((np.array(Tdfunc(Wnum), dtype=np.longdouble)[:, 0], (list(range(Ninter)), list(range(Ninter)))), shape=(Ninter, Ninter), dtype=np.longdouble)
                            Lij, Llu, Sterms = solve_def(Tnum=Tnum, Klambda=KLfunc(strain, Wnum), Klambda0=KLfunc(strain, Wnum0), Ndir=ndir,  Nspec=len(spec_list),
                                        Pref=(JumpPrefnum(strain,temp) / zfuncnum), L0=L0func(strain, Wnum), L=Llu,
                                        Site=[x(strain, Wnum) for x in SiteCorrFunc], sensi=(input_dataset.get('sensitivity') is not None), chol=chol)
                        else:
                            # Tnum is now T normalized by its own diagonal
                            Tdnum = sparse.csr_matrix((np.array(Tdfunc(Wnum), dtype=np.longdouble)[:, 0],
                                               (list(range(Ninter)), list(range(Ninter)))), shape=(Ninter, Ninter),
                                              dtype=np.longdouble)
                            Tnum = sparse.csr_matrix((np.array([1./a[0] for a in Tdfunc(Wnum)], dtype=np.longdouble),
                                               (list(range(Ninter)), list(range(Ninter)))), shape=(Ninter, Ninter),
                                              dtype=np.longdouble).dot(Tnum)

                            Lij, Sterms = solve_neumann(Tnum=Tnum, Klambda=KLfunc(strain, Wnum),
                                                         Klambda0=KLfunc(strain, Wnum0), Ndir=ndir,
                                                         Nspec=len(spec_list),
                                                         Pref=(JumpPrefnum(strain, temp) / zfuncnum),
                                                         L0=L0func(strain, Wnum),
                                                         Site=[x(strain, Wnum) for x in SiteCorrFunc],
                                                         sensi=(input_dataset.get('sensitivity') is not None),
                                                         Napprox=neumann_approx[0], alpha=neumann_approx[1], D=Tdnum)

                    ##################################################
                    for direction in range(ndir):
                        for a in range(len(spec_list)):
                            logger.info(("        [ {} ]").format(("{:21.14e} "*len(spec_list)).format(*Lij[direction][a])))
                        logger.info(" ")

                    if m == 'Lij':
                        with warn.catch_warnings():  # avoid printing "divide by zero" warning
                            warn.simplefilter("ignore")
                            for direction in range(ndir):
                                tLij = Lij[direction]
                                for sp1 in range(len(spec_list) - 1, -1, -1):
                                    if spec_list[sp1][1] == 0:
                                        tLij = np.delete(arr=tLij, axis=0, obj=sp1)
                                        tLij = np.delete(arr=tLij, axis=1, obj=sp1)
                                if np.all(tLij == 0): # Check if all elements are null
                                    logger.info('  !! All elements in matrix are null at T = {} K in direction {}.'.format(temp, direction))
                                    notthatdir.append(direction)
                                # Otherwise, check if matrix is symmetric
                                # The symmetry check is done by element-wise dividing the transpose Lij matrix by the original matrix, and comparing it element-wise to 1
                                elif not np.allclose(np.divide(tLij.T, tLij), np.ones(tLij.shape), atol=tol, rtol=tol):
                                    logger.info('  !! Cluster Onsager matrix is not perfectly symmetric in direction {} at T = {} K.'.format(direction, temp))
                                slow = [min([Lij[direction][e, e] for e in range(len(spec_list))]) for direction in range(ndir)]
                                tofile[direction].append([temp,strain*100,1000/temp,float(zfuncnum), Zfunc0, dissonum, np.float64(1.)/dissonum]+
                                         [Lij[direction][a, b] for a in range(len(spec_list)) for b in range(a,len(spec_list))]+
                                        #[Lij[direction][1, 0]] + [alldissonum[a] for a in alldissonum]) # temporary for ballistic
                                         [slow[0]]+ [alldissonum[a] for a in alldissonum]) # true line

                                # Printing additional results specified in the OUTOPTIONS tag: EX,UC,DR,CF,ND
                                for out in outopt:
                                    if outopt[out] and (out != "ND"):
                                        with open(outputfile+'_'+str(direction)+out+'.dat', 'a') as output:
                                            if out == 'DR': #drag ratios
                                                output.writelines((("{:^11.1f} {:^+15.5f} {:^16.5f}"+" {:s}".format(lij_digit_tag)*(koutdr-3))+"\n").format(temp, strain*100, 1000/temp, *flat_list([[np.divide(Lij[direction][b,a],Lij[direction][a,a]), np.divide(Lij[direction][a,b],Lij[direction][b,b])] for a in range(len(spec_list)) for b in range(a+1, len(spec_list))])))
                                            elif out == 'CF': #correlation coefficient
                                                output.writelines((("{:^11.1f} {:^+15.5f} {:^16.5f}"+" {:s}".format(lij_digit_tag)*(kout-3))+"\n").format(temp, strain*100, 1000/temp, *[np.divide(Lij[direction][ a, b], float(JumpPrefnum(e, temp)*L0func(strain, Wnum)[direction][a][b]/zfuncnum)) for a in range(len(spec_list)) for b in range(a, len(spec_list))]))
                                            elif out == 'UC': #uncorrelated coefficient
                                                output.writelines((("{:^11.1f} {:^+15.5f} {:^16.5f}"+" {:s}".format(lij_digit_tag)*(kout-3))+"\n").format(temp, strain*100, 1000/temp, *[float(JumpPrefnum(e, temp)*L0func(strain, Wnum)[direction][a][b]/zfuncnum) for a in range(len(spec_list)) for b in range(a,len(spec_list))]))

                        if len(kiraloop) > 0:
                            with sym.evaluate(False):
                                convergence_analysis(kiraloop=kiraloop, maxkira=copy.deepcopy(maxkira), Wnum=Wnum,
                                                     Cnum=Cnum, z=zfuncnum,
                                                     z0=Zfunc0, prec=input_dataset.get('precision'),
                                                     L0=L0func(strain, Wnum),
                                                     JP=JumpPrefnum(strain, temp), strain=strain, temp=temp, ndir=ndir,
                                                     nsp=len(spec_list), T=Tnum, KL=KLfunc(strain, Wnum),
                                                     KL0=KLfunc(strain, Wnum0),
                                                     Td=Tdfunc(Wnum),
                                                     directory=dir + 'CONVERGENCE_STUDY_{:.0f}/'.format(fileN), refLij=Lij, 
						     lij_digit_tag=lij_digit_tag, lij_title_digit_tag=lij_title_digit_tag, chol=chol)

                        if input_dataset.get('sensitivity') is not None:
                            sensitivity_study(KL=[np.matrix(KLfunc(Smax, Wnum)[d], dtype=float) for d in range(0,ndir)],
                                              L=Llu, temp="T{:.0f}_S{:.4f}".format(temp,strain), dir=dir, W=np.array([Wnum[a] for a in uniqueJF], dtype=float),
                                              dL0=derL0, dKL=derKL, dT=derT, uniqueJF=uniqueJF, derS=derSiteCorrFunc, Sterms=Sterms,
                                              directions=[n for n in range(int(input_dataset['sensitivity'][0])) if n not in notthatdir])

                    elif m == 'mob':
                        #  Print exchange coefficients if needed
                        if outopt['EX']:
                            for direction in range(0,ndir):
                                with open(outputfile + '_' + str(direction) + 'EX.dat', 'a') as output:
                                    output.writelines((("{:^11.1f} {:^+15.5f} {:^16.5f}"+" {:s}".format(lij_digit_tag)*(kout-3)) + "\n").format(temp, strain*100, 1000/temp, *[(a-Lij[direction][0,0]) for a in tofile[direction][k][7:-1]]))
                        # remove coefficients with no components of this species
                        nzspeclist = []
                        for sp1 in range(len(spec_list)-1,-1,-1):
                            if spec_list[sp1][1] == 0:
                                Lij[direction] = np.delete(arr=Lij[direction], axis=0, obj=sp1)
                                Lij[direction] = np.delete(arr=Lij[direction], axis=1, obj=sp1)
                            else:
                                nzspeclist.append(spec_list[sp1])
                        nzspeclist = list(reversed(nzspeclist))
                        # divide mobility coefficients by number of species
                        for direction in range(ndir):
                            for sp1 in range(len(nzspeclist)):
                                for sp2 in range(len(nzspeclist)):
                                    Lij[direction][sp1,sp2] = Lij[direction][sp1,sp2] / (nzspeclist[sp1][1]*nzspeclist[sp2][1])
                        # check if mobility is scalar
                        for direction in range(ndir):
                            if not np.allclose(Lij[direction]/Lij[direction][0,0], np.ones((len(nzspeclist),len(nzspeclist))), atol=tol, rtol=tol):
                                logger.info('!! Mobility coefficient is not a scalar in direction {}'.format(direction))
                            with warn.catch_warnings():  # avoid printing "divide by zero" warning
                                warn.simplefilter("ignore")
                                tofile[direction][k] = tofile[direction][k][:-(len(alldissonum)+1)]+[Lij[direction][0,0],
                                                        np.divide(Lij[direction][0,0], tofile[direction][k][-1]),
                                                        np.sqrt(2*Lij[direction][0,0]/tofile[direction][k][5])]+tofile[direction][k][len(tofile[direction][k])-len(alldissonum):]
                        k+=1

        #print output file
        for direction in range(ndir):
            with open(outputfile+'_'+str(direction)+'.dat', 'w') as output:
                ncoeff = int(0.5*len(spec_list)*(len(spec_list)+1))
                if mode_dict["mob"]:
                    output.writelines(("#{:^10s} {:^15s} {:^16s} {:^17s} {:^10s}"+(" {:s}".format(lij_title_digit_tag)*(5+ncoeff+len(alldissonum)))+"\n").format(
                        '1) T [K]', '2) Strain [%]', '3) 1000/T [/K]', '4) Z', '5) Z0', '6) Diss [/s]', '7) Lifetime [s]',
                        *[str(8+b+sum([(len(spec_list)-e) for e in range(1,a+1)]))+') L_'+str(a)+str(b) for a in range(0, len(spec_list)) for b in range(a, len(spec_list))],
                        str(ncoeff+8)+') M', str(ncoeff+9)+') Conv', str(ncoeff+10)+') MFP',
                        *[str(ncoeff+11+idx)+') D'+disso.replace('_','') for idx, disso in enumerate(alldissonum)]))
                else:
                    output.writelines(("#{:^10s} {:^15s} {:^16s} {:^17s} {:^10s}"+(" {:s}".format(lij_title_digit_tag)*(3+ncoeff+len(alldissonum)))+"\n").format(
                        '1) T [K]', '2) Strain [%]', '3) 1000/T [/K]', '4) Z', '5) Z0', '6) Diss [/s]', '7) Lifetime [s]',
                        *[str(8+b+sum([(len(spec_list)-e) for e in range(1,a+1)]))+') L_'+str(a)+str(b) for a in range(0, len(spec_list)) for b in range(a, len(spec_list))],
                        str(ncoeff+8)+') L_slowest',
                        *[str(ncoeff+9+idx)+') D'+disso.replace('_','') for idx, disso in enumerate(alldissonum)]))
                for res in tofile[direction]:
                    output.writelines(("{:^11.1f} {:^+15.5f} {:^16.5f} {:^17.6E} {:^10.0f}"+(" {:s}".format(lij_digit_tag))*(len(tofile[0][0])-5)+"\n").format(*res))
            # Plotting transport coefficients in temperature-strain plane if non-zero strain
            if len(strain_list) > 1 and len(temperature_list) > 1 and (not np.all(Lij[direction] == 0)) and False: # false added because of plotting issue with monosia under strain
                pp = PdfPages(dir + 'TempStrain_{}.pdf'.format(direction))
                ksp = -1
                for sp1 in range(len(spec_list)):
                    for sp2 in range(sp1, len(spec_list)):
                        ksp += 1
                        data = np.zeros((len(temperature_list),len(strain_list)))
                        Xval = np.zeros((len(temperature_list),len(strain_list)))
                        Yval = np.zeros((len(temperature_list),len(strain_list)))
                        for y in range(0, len(strain_list)):
                            for x in range(0, len(temperature_list)):
                                Xval[x, y] = temperature_list[x]
                                Yval[x, y] = strain_list[y]*100
                                data[x, y] = tofile[direction][len(strain_list)*x + y][7+ksp]
                        fig, ax = plt.subplots()
                        if np.abs(data).min() != 0 and np.abs(data).max() != 0:
                            #pcm = ax.pcolormesh(Xval, Yval, np.abs(data), norm=colors.LogNorm(vmin=10**int(np.floor(np.log10(np.abs(data).min()))),
                            #                                                               vmax=10**int(np.ceil(np.log10(np.abs(data).max())))), cmap='jet', shading='gouraud')
                            pcm = ax.pcolormesh(Xval, Yval, np.abs(data),
                                          norm=colors.LogNorm(vmin=10**-42, vmax=10**-12), cmap='jet', shading='gouraud')
                        if data.min() < 0 and data.max() > 0 :
                            plt.contour(Xval, Yval, data, levels=[0], colors='k', linewidths=2, linestyles='solid')
                        plt.contour(Xval, Yval, np.abs(data), levels=[10**a for a in range(int(np.floor(np.min(np.log10(np.abs(data))))),
                                        int(np.ceil(np.max(np.log10(np.abs(data))))),2)], colors='k', linewidths=1, linestyles='dashed')
                        ax.grid(color='k', linestyle='-', linewidth=0.3)
                        fig.colorbar(pcm, ax=ax, extend='max', ticks=LogLocator(subs=range(0)))
                        pcm.colorbar.set_label('$|L_{}{}|$  (${}$)'.format("{"+spec_list[sp1][0][0],spec_list[sp2][0][0]+"}", units.replace("2","^2")), fontsize=12)
                        ax.set_xlabel('Temperature  (K)', fontsize=12)
                        ax.set_ylabel('Strain  (%)', fontsize=12)
                        pp.savefig()
                if outopt['DR']: # plot drag ratios
                    for sp1 in range(len(spec_list)):
                        ii_idx = int(sp1*(len(spec_list)-0.5*(sp1-1)))  # index of diagonal coefficient
                        for sp2 in range(len(spec_list)):
                            if sp1 != sp2:
                                ij_idx = int(np.min([sp1, sp2])*(len(spec_list)-0.5*(np.min([sp1, sp2])-1)) + np.abs(sp1-sp2))
                                data = np.zeros((len(temperature_list), len(strain_list)))
                                Xval = np.zeros((len(temperature_list), len(strain_list)))
                                Yval = np.zeros((len(temperature_list), len(strain_list)))
                                for y in range(0, len(strain_list)):
                                    for x in range(0, len(temperature_list)):
                                        Xval[x, y] = temperature_list[x]
                                        Yval[x, y] = strain_list[y] * 100
                                        data[x, y] = tofile[direction][len(strain_list) * x + y][7 + ij_idx]/tofile[direction][len(strain_list) * x + y][7 + ii_idx]
                                fig, ax = plt.subplots()
                                if np.abs(data).min() != 0 and np.abs(data).max() != 0:
                                    pcm = ax.pcolormesh(Xval, Yval, data, cmap='jet', shading='gouraud',
                                            norm=colors.Normalize(vmin=np.floor(data.min()), vmax=np.ceil(data.max())))
                                if data.min() < 0 and data.max() > 0:
                                    plt.contour(Xval, Yval, data, levels=[0], colors='k', linewidths=2, linestyles='solid')
                                plt.contour(Xval, Yval, data,
                                            levels=list(np.arange(int(np.floor(data.min())), int(np.ceil(data.max())),
                                                                0.5)), colors='k', linewidths=1,  linestyles='dashed')
                                ax.grid(color='k', linestyle='-', linewidth=0.3)
                                fig.colorbar(pcm, ax=ax, extend='max')
                                pcm.colorbar.set_label(
                                    '$|L_{}{}/L_{}{}|$  (${}$)'.format("{" + spec_list[sp2][0][0], spec_list[sp1][0][0] + "}",
                                                                       "{" + spec_list[sp1][0][0], spec_list[sp1][0][0] + "}",
                                                                units.replace("2", "^2")), fontsize=12)
                                ax.set_xlabel('Temperature  (K)', fontsize=12)
                                ax.set_ylabel('Strain  (%)', fontsize=12)
                                pp.savefig()
                if mode_dict['mob']:
                    ksp += 1
                    data = np.zeros((len(temperature_list), len(strain_list)))
                    Xval = np.zeros((len(temperature_list), len(strain_list)))
                    Yval = np.zeros((len(temperature_list), len(strain_list)))
                    for y in range(0, len(strain_list)):
                        for x in range(0, len(temperature_list)):
                            Xval[x, y] = temperature_list[x]
                            Yval[x, y] = strain_list[y] * 100
                            data[x, y] = tofile[direction][len(temperature_list) * y + x][7 + ksp]
                    fig, ax = plt.subplots()
                    if data.min() > 0 and data.max() > 0:
                        pcm = ax.pcolormesh(Xval, Yval, np.abs(data), norm=colors.LogNorm(vmin=10 ** int(np.floor(np.log10(data.min()))),
                                                                vmax=10 ** int(np.ceil(np.log10(data.max())))), cmap='jet', shading='gouraud')
                    plt.contour(Xval, Yval, np.abs(data), levels=[10**a for a in range(int(np.floor(np.min(np.log10(np.abs(data))))),
                                        int(np.ceil(np.max(np.log10(np.abs(data))))),2)], colors='k', linewidths=1,linestyles='dashed')
                    ax.grid(color='k', linestyle='-', linewidth=0.3)
                    fig.colorbar(pcm, ax=ax, extend='max', ticks=LogLocator(subs=range(0)))
                    pcm.colorbar.set_label('Mobility  (${}$)'.format(units.replace("2","^2")), fontsize=12)
                    ax.set_xlabel('Temperature  (K)', fontsize=12)
                    ax.set_ylabel('Strain  (%)', fontsize=12)
                    pp.savefig()
                pp.close()
    del crystal_name, kra_sp, units_dict

    # Copy numerical input file into result directory
    copy_input_path = dir + myinput.split("/")[-1]
    if myinput[-4:] != ".txt":
        copy_input_path += ".txt"
    if input_dataset['directory'] == ["cwd"]:
        if not os.path.exists(copy_input_path):  # necessary when using the 'cwd' directory option
            copyfile(myinput, copy_input_path)
    else:
        copyfile(myinput, copy_input_path)

    # Stop measuring execution time and print elapsed time
    stop_time = tm.time()
    logger.info("Execution time: {:.3f} s.".format(stop_time-start_time))
    logger.info("Peak memory usage: {:.3f} MB.".format(process_for_memory_tracking.memory_info()[0] * 1e-6))  # peak memory usage

    # END CODE-----------------------------------


if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 2:
        raise produce_error_and_quit("!! Incorrect call of {}. Correct usage: << ./kineclue_num input_file_path >>".format(sys.argv[0]))
    else:
        numrun(sys.argv[1]) # actually running the code        
