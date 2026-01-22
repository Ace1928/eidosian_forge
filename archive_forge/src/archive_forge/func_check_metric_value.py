import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
def check_metric_value(self, args, state, control, metric_value):
    operator = np.greater if args.greater_is_better else np.less
    if state.best_metric is None or (operator(metric_value, state.best_metric) and abs(metric_value - state.best_metric) > self.early_stopping_threshold):
        self.early_stopping_patience_counter = 0
    else:
        self.early_stopping_patience_counter += 1