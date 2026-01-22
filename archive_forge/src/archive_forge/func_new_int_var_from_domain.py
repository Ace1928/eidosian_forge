import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def new_int_var_from_domain(self, domain: Domain, name: str) -> IntVar:
    """Create an integer variable from a domain.

        A domain is a set of integers specified by a collection of intervals.
        For example, `model.new_int_var_from_domain(cp_model.
             Domain.from_intervals([[1, 2], [4, 6]]), 'x')`

        Args:
          domain: An instance of the Domain class.
          name: The name of the variable.

        Returns:
            a variable whose domain is the given domain.
        """
    return IntVar(self.__model, domain, name)