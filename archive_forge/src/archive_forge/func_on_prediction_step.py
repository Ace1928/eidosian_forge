import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
    if state.is_world_process_zero and has_length(eval_dataloader):
        if self.prediction_bar is None:
            self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True)
        self.prediction_bar.update(1)