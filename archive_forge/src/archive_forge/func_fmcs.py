import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def fmcs(mols, minNumAtoms=2, maximize=Default.maximize, atomCompare=Default.atomCompare, bondCompare=Default.bondCompare, threshold=1.0, matchValences=Default.matchValences, ringMatchesRingOnly=False, completeRingsOnly=False, timeout=Default.timeout, times=None, verbose=False, verboseDelay=1.0):
    timer = Timer()
    timer.mark('start fmcs')
    if minNumAtoms < 2:
        raise ValueError('minNumAtoms must be at least 2')
    if timeout is not None:
        if timeout <= 0.0:
            raise ValueError('timeout must be None or a positive value')
    threshold_count = _get_threshold_count(len(mols), threshold)
    if threshold_count > len(mols):
        return MCSResult(-1, -1, None, 1)
    if completeRingsOnly:
        ringMatchesRingOnly = True
    try:
        atom_typer = atom_typers[atomCompare]
    except KeyError:
        raise ValueError('Unknown atomCompare option %r' % (atomCompare,))
    try:
        bond_typer = bond_typers[bondCompare]
    except KeyError:
        raise ValueError('Unknown bondCompare option %r' % (bondCompare,))
    typed_mols = convert_input_to_typed_molecules(mols, atom_typer, bond_typer, matchValences=matchValences, ringMatchesRingOnly=ringMatchesRingOnly)
    bondtype_counts = get_canonical_bondtype_counts(typed_mols)
    supported_bondtypes = set()
    for bondtype, count_list in bondtype_counts.items():
        if len(count_list) >= threshold_count:
            supported_bondtypes.add(bondtype)
    fragmented_mols = [remove_unknown_bondtypes(typed_mol, bondtype_counts) for typed_mol in typed_mols]
    timer.mark('end fragment')
    sizes = []
    max_num_atoms = fragmented_mols[0].rdmol.GetNumAtoms()
    max_num_bonds = fragmented_mols[0].rdmol.GetNumBonds()
    ignored_count = 0
    for tiebreaker, (typed_mol, fragmented_mol) in enumerate(zip(typed_mols, fragmented_mols)):
        num_atoms, num_bonds = find_upper_fragment_size_limits(fragmented_mol.rdmol, fragmented_mol.rdmol_atoms)
        if num_atoms < minNumAtoms:
            ignored_count += 1
            if ignored_count + threshold_count > len(mols):
                timer.mark('end select')
                timer.mark('end fmcs')
                _update_times(timer, times)
                return MCSResult(-1, -1, None, True)
        else:
            if num_atoms < max_num_atoms:
                max_num_atoms = num_atoms
            if num_bonds < max_num_bonds:
                max_num_bonds = num_bonds
            sizes.append((num_bonds, num_atoms, tiebreaker, typed_mol, fragmented_mol))
    if len(sizes) < threshold_count:
        timer.mark('end select')
        timer.mark('end fmcs')
        _update_times(timer, times)
        return MCSResult(-1, -1, None, True)
    assert min((size[1] for size in sizes)) >= minNumAtoms
    sizes.sort()
    timer.mark('end select')
    fragmented_mols = [size_info[4] for size_info in sizes]
    typed_mols = [size_info[3].rdmol for size_info in sizes]
    timer.mark('start enumeration')
    mcs_result = compute_mcs(fragmented_mols, typed_mols, minNumAtoms, threshold_count=threshold_count, maximize=maximize, completeRingsOnly=completeRingsOnly, timeout=timeout, timer=timer, verbose=verbose, verboseDelay=verboseDelay)
    timer.mark('end fmcs')
    _update_times(timer, times)
    return mcs_result