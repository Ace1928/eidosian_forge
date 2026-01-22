from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from .beam_constraints import Constraint, ConstraintListState
def check_completes_constraints(self, sequence):
    new_state = self.make_constraint_states(1)[0]
    new_state.reset(sequence)
    return new_state.completed