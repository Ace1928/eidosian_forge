import os
import tempfile
from typing import Callable, List, Optional
from uuid import uuid4
import cloudpickle
from fs.base import FS as FSBase
from triad import FileSystem
from tune.concepts.checkpoint import Checkpoint
from tune.concepts.flow import Monitor, Trial, TrialDecision, TrialJudge, TrialReport
def run_single_iteration(self) -> TrialReport:
    raise NotImplementedError