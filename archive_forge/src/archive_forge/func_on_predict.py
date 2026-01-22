import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
def on_predict(self, args, state, control, **kwargs):
    if state.is_world_process_zero:
        if self.prediction_bar is not None:
            self.prediction_bar.close()
        self.prediction_bar = None