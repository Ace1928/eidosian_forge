from typing import Any, Callable, Optional
from tune._utils import run_monitored_process
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.constants import TUNE_STOPPER_DEFAULT_CHECK_INTERVAL
@property
def distributable(self) -> bool:
    return True