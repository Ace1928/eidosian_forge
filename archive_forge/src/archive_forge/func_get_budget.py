from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
def get_budget(self, trial: Trial, rung: int) -> float:
    return 0.0