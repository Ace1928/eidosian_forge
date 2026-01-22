import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
    return self.call_event('on_substep_end', args, state, control)