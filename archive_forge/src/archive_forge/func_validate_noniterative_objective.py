from typing import Any, Callable, Optional
from tune._utils import run_monitored_process
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
def validate_noniterative_objective(func: NonIterativeObjectiveFunc, trial: Trial, validator: Callable[[TrialReport], None], optimizer: Optional[NonIterativeObjectiveLocalOptimizer]=None, logger: Any=None) -> None:
    _optimizer = optimizer or NonIterativeObjectiveLocalOptimizer()
    validator(_optimizer.run_monitored_process(func, trial, lambda: False, interval='1sec', logger=logger))