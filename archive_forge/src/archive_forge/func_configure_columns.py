import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def configure_columns(self, trainer: 'pl.Trainer') -> list:
    return [TextColumn('[progress.description]{task.description}'), CustomBarColumn(complete_style=self.theme.progress_bar, finished_style=self.theme.progress_bar_finished, pulse_style=self.theme.progress_bar_pulse), BatchesProcessedColumn(style=self.theme.batch_progress), CustomTimeColumn(style=self.theme.time), ProcessingSpeedColumn(style=self.theme.processing_speed)]