from typing import Any, Callable, List, Optional, Union
from fugue import FugueWorkflow
from triad import assert_or_throw, conditional_dispatcher
from tune.concepts.dataset import TuneDataset, TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Space
from tune.concepts.logger import MetricLogger
from tune.constants import (
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
@conditional_dispatcher(entry_point=TUNE_PLUGINS)
def parse_noniterative_objective(obj: Any) -> NonIterativeObjectiveFunc:
    assert_or_throw(obj is not None, TuneCompileError("objective can't be None"))
    if isinstance(obj, NonIterativeObjectiveFunc):
        return obj
    if callable(obj):
        return to_noniterative_objective(obj)
    raise TuneCompileError(f'{obj} is not a valid non iterative objective function')