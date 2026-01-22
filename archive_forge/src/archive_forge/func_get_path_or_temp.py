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
def get_path_or_temp(self, path: str) -> str:
    if path is None or path == '':
        path = self._tmp
    assert_or_throw(path != '', TuneCompileError('path or temp path must be set'))
    return path