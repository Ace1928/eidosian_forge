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
def group_settings_greedy(tomo_expt: Experiment) -> Experiment:
    """
    Greedy method to group ExperimentSettings in a given Experiment

    :param tomo_expt: Experiment to group ExperimentSettings within
    :return: Experiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = _max_tpb_overlap(tomo_expt)
    grouped_expt_settings_list = list(diag_sets.values())
    grouped_tomo_expt = Experiment(grouped_expt_settings_list, program=tomo_expt.program, symmetrization=tomo_expt.symmetrization)
    return grouped_tomo_expt