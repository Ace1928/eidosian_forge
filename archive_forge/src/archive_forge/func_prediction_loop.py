import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool]=None, ignore_keys: Optional[List[str]]=None, metric_key_prefix: str='eval') -> EvalLoopOutput:
    """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
    args = self.args
    if not has_length(dataloader):
        raise ValueError('dataloader must implement a working __len__')
    prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
    if self.is_deepspeed_enabled and self.deepspeed is None:
        _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
    model = self._wrap_model(self.model, training=False, dataloader=dataloader)
    if len(self.accelerator._models) == 0 and model is self.model:
        model = self.accelerator.prepare(model) if self.is_deepspeed_enabled else self.accelerator.prepare_model(model, evaluation_mode=True)
        if self.is_fsdp_enabled:
            self.model = model
        if model is not self.model:
            self.model_wrapped = model
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped
    if not self.is_in_train:
        if args.fp16_full_eval:
            model = model.to(dtype=torch.float16, device=args.device)
        elif args.bf16_full_eval:
            model = model.to(dtype=torch.bfloat16, device=args.device)
    batch_size = dataloader.batch_size
    num_examples = self.num_examples(dataloader)
    logger.info(f'***** Running {description} *****')
    logger.info(f'  Num examples = {num_examples}')
    logger.info(f'  Batch size = {batch_size}')
    losses_host: torch.Tensor = None
    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
    inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None
    world_size = max(1, args.world_size)
    eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
    if not prediction_loss_only:
        make_multiple_of = None
        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, SequentialDistributedSampler):
            make_multiple_of = dataloader.sampler.batch_size
        preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
        labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
        inputs_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
    model.eval()
    if args.past_index >= 0:
        self._past = None
    self.callback_handler.eval_dataloader = dataloader
    for step, inputs in enumerate(dataloader):
        loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        main_input_name = getattr(self.model, 'main_input_name', 'input_ids')
        inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None
        if loss is not None:
            losses = loss.repeat(batch_size)
            losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
        if logits is not None:
            preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
        if labels is not None:
            labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
        if inputs_decode is not None:
            inputs_host = inputs_decode if inputs_host is None else nested_concat(inputs_host, inputs_decode, padding_index=-100)
        self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
        if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
            eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, 'eval_losses'))
            if not prediction_loss_only:
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, 'eval_preds'))
                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, 'eval_label_ids'))
                inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, 'eval_inputs_ids'))
            losses_host, preds_host, labels_host, inputs_host = (None, None, None, None)
    if args.past_index and hasattr(self, '_past'):
        delattr(self, '_past')
    eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, 'eval_losses'))
    if not prediction_loss_only:
        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, 'eval_preds'))
        labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, 'eval_label_ids'))
        inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, 'eval_inputs_ids'))
    eval_loss = eval_losses_gatherer.finalize()
    preds = preds_gatherer.finalize() if not prediction_loss_only else None
    label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
    inputs_ids = inputs_gatherer.finalize() if not prediction_loss_only else None
    if self.compute_metrics is not None and preds is not None and (label_ids is not None):
        if args.include_inputs_for_metrics:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids))
        else:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
    else:
        metrics = {}
    metrics = denumpify_detensorize(metrics)
    if eval_loss is not None:
        metrics[f'{metric_key_prefix}_loss'] = eval_loss.mean().item()
    for key in list(metrics.keys()):
        if not key.startswith(f'{metric_key_prefix}_'):
            metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key)
    return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics=metrics, num_samples=num_examples)