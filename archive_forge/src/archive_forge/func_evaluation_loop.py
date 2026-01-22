import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import (
def evaluation_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool]=None, ignore_keys: Optional[List[str]]=None, metric_key_prefix: str='eval') -> EvalLoopOutput:
    """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
    if self.generate_during_eval:
        num_samples = len(dataloader.dataset)
        random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)
        random_batch_dataset = dataloader.dataset.select(random_indices)
        random_batch = self.data_collator(random_batch_dataset)
        random_batch = self._prepare_inputs(random_batch)
        policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)
        self.log({'game_log': wandb.Table(columns=['Prompt', 'Policy', 'Ref Model'], rows=[[prompt, pol[len(prompt):], ref[len(prompt):]] for prompt, pol, ref in zip(random_batch['prompt'], policy_output_decoded, ref_output_decoded)])})
        self.state.log_history.pop()
    initial_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
    return initial_output