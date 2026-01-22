import copy
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check
from fugue._utils.interfaceless import is_class_method
from triad import assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.convert import get_caller_global_local_vars, to_function
from tune.concepts.flow import Trial, TrialReport
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc
def noniterative_objective(func: Optional[Callable]=None, min_better: bool=True) -> Callable[[Any], NonIterativeObjectiveFunc]:

    def deco(func: Callable) -> NonIterativeObjectiveFunc:
        assert_or_throw(not is_class_method(func), NotImplementedError("non_iterative_objective decorator can't be used on class methods"))
        return _NonIterativeObjectiveFuncWrapper.from_func(func, min_better)
    if func is None:
        return deco
    else:
        return deco(func)