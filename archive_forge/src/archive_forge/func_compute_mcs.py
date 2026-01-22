import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def compute_mcs(fragmented_mols, typed_mols, minNumAtoms, threshold_count=None, maximize=Default.maximize, completeRingsOnly=Default.completeRingsOnly, timeout=Default.timeout, timer=None, verbose=False, verboseDelay=1.0):
    assert timer is not None
    assert 0 < threshold_count <= len(fragmented_mols), threshold_count
    assert len(fragmented_mols) == len(typed_mols)
    assert len(fragmented_mols) >= 2
    if threshold_count is None:
        threshold_count = len(fragmented_mols)
    else:
        assert threshold_count >= 2, threshold_count
    atom_assignment = Uniquer()
    if verbose:
        if verboseDelay < 0.0:
            raise ValueError('verboseDelay may not be negative')
        matches_all_targets = VerboseCachingTargetsMatcher(typed_mols[1:], threshold_count - 1)
        heapops = VerboseHeapOps(matches_all_targets.report, verboseDelay)
        push = heapops.heappush
        pop = heapops.heappop
        end_verbose = heapops.trigger_report
    else:
        matches_all_targets = CachingTargetsMatcher(typed_mols[1:], threshold_count - 1)
        push = heappush
        pop = heappop
        end_verbose = lambda: 1
    try:
        prune, hits_class = _maximize_options[maximize, bool(completeRingsOnly)]
    except KeyError:
        raise ValueError("Unknown 'maximize' option %r" % (maximize,))
    hits = hits_class(timer, verbose)
    remaining_time = None
    if timeout is not None:
        stop_time = time.perf_counter() + timeout
    for query_index, fragmented_query_mol in enumerate(fragmented_mols):
        enumerated_query_fragments = fragmented_mol_to_enumeration_mols(fragmented_query_mol, minNumAtoms)
        targets = typed_mols
        if timeout is not None:
            remaining_time = stop_time - time.perf_counter()
        success = enumerate_subgraphs(enumerated_query_fragments, prune, atom_assignment, matches_all_targets, hits, remaining_time, push, pop)
        if query_index + threshold_count >= len(fragmented_mols):
            break
        if not success:
            break
        matches_all_targets.shift_targets()
    end_verbose()
    result = hits.get_result(success)
    if result.num_atoms < minNumAtoms:
        return MCSResult(-1, -1, None, result.completed)
    return result