import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError
from . import __version__
from .models.auto.modeling_auto import (
from .training_args import ParallelMode
from .utils import (
def parse_log_history(log_history):
    """
    Parse the `log_history` of a Trainer to get the intermediate and final evaluation results.
    """
    idx = 0
    while idx < len(log_history) and 'train_runtime' not in log_history[idx]:
        idx += 1
    if idx == len(log_history):
        idx -= 1
        while idx >= 0 and 'eval_loss' not in log_history[idx]:
            idx -= 1
        if idx >= 0:
            return (None, None, log_history[idx])
        else:
            return (None, None, None)
    train_log = log_history[idx]
    lines = []
    training_loss = 'No log'
    for i in range(idx):
        if 'loss' in log_history[i]:
            training_loss = log_history[i]['loss']
        if 'eval_loss' in log_history[i]:
            metrics = log_history[i].copy()
            _ = metrics.pop('total_flos', None)
            epoch = metrics.pop('epoch', None)
            step = metrics.pop('step', None)
            _ = metrics.pop('eval_runtime', None)
            _ = metrics.pop('eval_samples_per_second', None)
            _ = metrics.pop('eval_steps_per_second', None)
            _ = metrics.pop('eval_jit_compilation_time', None)
            values = {'Training Loss': training_loss, 'Epoch': epoch, 'Step': step}
            for k, v in metrics.items():
                if k == 'eval_loss':
                    values['Validation Loss'] = v
                else:
                    splits = k.split('_')
                    name = ' '.join([part.capitalize() for part in splits[1:]])
                    values[name] = v
            lines.append(values)
    idx = len(log_history) - 1
    while idx >= 0 and 'eval_loss' not in log_history[idx]:
        idx -= 1
    if idx > 0:
        eval_results = {}
        for key, value in log_history[idx].items():
            if key.startswith('eval_'):
                key = key[5:]
            if key not in ['runtime', 'samples_per_second', 'steps_per_second', 'epoch', 'step']:
                camel_cased_key = ' '.join([part.capitalize() for part in key.split('_')])
                eval_results[camel_cased_key] = value
        return (train_log, lines, eval_results)
    else:
        return (train_log, lines, None)