from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import do_nothing_closure
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.optimization.closure import OutputResult
from pytorch_lightning.loops.progress import _Progress, _ReadyCompletedTracker
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
@classmethod
def from_training_step_output(cls, training_step_output: STEP_OUTPUT) -> 'ManualResult':
    extra = {}
    if isinstance(training_step_output, dict):
        extra = training_step_output.copy()
    elif isinstance(training_step_output, Tensor):
        extra = {'loss': training_step_output}
    elif training_step_output is not None:
        raise MisconfigurationException('In manual optimization, `training_step` must either return a Tensor or have no return.')
    if 'loss' in extra:
        extra['loss'] = extra['loss'].detach()
    return cls(extra=extra)