import os
import tempfile
from typing import Callable, List, Optional
from uuid import uuid4
import cloudpickle
from fs.base import FS as FSBase
from triad import FileSystem
from tune.concepts.checkpoint import Checkpoint
from tune.concepts.flow import Monitor, Trial, TrialDecision, TrialJudge, TrialReport
@property
def current_trial(self) -> Trial:
    assert self._current_trial is not None
    return self._current_trial