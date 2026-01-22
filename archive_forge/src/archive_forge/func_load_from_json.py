import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
@classmethod
def load_from_json(cls, json_path: str):
    """Create an instance from the content of `json_path`."""
    with open(json_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return cls(**json.loads(text))