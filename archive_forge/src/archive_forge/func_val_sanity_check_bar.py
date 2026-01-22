import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
@property
def val_sanity_check_bar(self) -> Task:
    assert self.progress is not None
    assert self.val_sanity_progress_bar_id is not None
    return self.progress.tasks[self.val_sanity_progress_bar_id]