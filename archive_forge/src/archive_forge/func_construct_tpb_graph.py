import functools
import itertools
from operator import mul
from typing import Dict, List, Iterable, Sequence, Set, Tuple, Union, cast
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, TensorProductState, _OneQState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.paulis import PauliTerm, sI
from pyquil.quil import Program
def construct_tpb_graph(experiments: Experiment) -> nx.Graph:
    """
    Construct a graph where an edge signifies two experiments are diagonal in a TPB.
    """
    g = nx.Graph()
    for expt in experiments:
        assert len(expt) == 1, 'already grouped?'
        unpacked_expt = expt[0]
        if unpacked_expt not in g:
            g.add_node(unpacked_expt, count=1)
        else:
            g.nodes[unpacked_expt]['count'] += 1
    for expt1, expt2 in itertools.combinations(experiments, r=2):
        unpacked_expt1 = expt1[0]
        unpacked_expt2 = expt2[0]
        if unpacked_expt1 == unpacked_expt2:
            continue
        max_weight_in = _max_weight_state([unpacked_expt1.in_state, unpacked_expt2.in_state])
        max_weight_out = _max_weight_operator([unpacked_expt1.out_operator, unpacked_expt2.out_operator])
        if max_weight_in is not None and max_weight_out is not None:
            g.add_edge(unpacked_expt1, unpacked_expt2)
    return g