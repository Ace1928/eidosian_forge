from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
def prepare_tf_dataset(self, dataset: 'datasets.Dataset', batch_size: int=8, shuffle: bool=True, tokenizer: Optional['PreTrainedTokenizerBase']=None, collate_fn: Optional[Callable]=None, collate_fn_args: Optional[Dict[str, Any]]=None, drop_remainder: Optional[bool]=None, prefetch: bool=True):
    """
        Wraps a HuggingFace [`~datasets.Dataset`] as a `tf.data.Dataset` with collation and batching. This method is
        designed to create a "ready-to-use" dataset that can be passed directly to Keras methods like `fit()` without
        further modification. The method will drop columns from the dataset if they don't match input names for the
        model. If you want to specify the column names to return rather than using the names that match this model, we
        recommend using `Dataset.to_tf_dataset()` instead.

        Args:
            dataset (`Any`):
                A [~`datasets.Dataset`] to be wrapped as a `tf.data.Dataset`.
            batch_size (`int`, defaults to 8):
                The size of batches to return.
            shuffle (`bool`, defaults to `True`):
                Whether to return samples from the dataset in random order. Usually `True` for training datasets and
                `False` for validation/test datasets.
            tokenizer ([`PreTrainedTokenizerBase`], *optional*):
                A `PreTrainedTokenizer` that will be used to pad samples to create batches. Has no effect if a specific
                `collate_fn` is passed instead.
            collate_fn (`Callable`, *optional*):
                A function that collates samples from the dataset into a single batch. Defaults to
                `DefaultDataCollator` if no `tokenizer` is supplied or `DataCollatorWithPadding` if a `tokenizer` is
                passed.
            collate_fn_args (`Dict[str, Any]`, *optional*):
                A dict of arguments to pass to the `collate_fn` alongside the list of samples.
            drop_remainder (`bool`, *optional*):
                Whether to drop the final batch, if the batch_size does not evenly divide the dataset length. Defaults
                to the same setting as `shuffle`.
            prefetch (`bool`, defaults to `True`):
                Whether to add prefetching to the end of the `tf.data` pipeline. This is almost always beneficial for
                performance, but can be disabled in edge cases.


        Returns:
            `Dataset`: A `tf.data.Dataset` which is ready to pass to the Keras API.
        """
    requires_backends(self, ['datasets'])
    import datasets
    if collate_fn is None:
        if tokenizer is None:
            collate_fn = DefaultDataCollator(return_tensors='np')
        else:
            collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='np')
    if collate_fn_args is None:
        collate_fn_args = {}
    if not isinstance(dataset, datasets.Dataset):
        raise TypeError('Dataset argument should be a datasets.Dataset!')
    model_inputs = list(inspect.signature(self.call).parameters)
    model_labels = find_labels(self.__class__)
    if 'cols_to_retain' in list(inspect.signature(dataset._get_output_signature).parameters.keys()):
        output_signature, _ = dataset._get_output_signature(dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args, cols_to_retain=model_inputs)
    else:
        unwanted_columns = [feature for feature in dataset.features if feature not in model_inputs and feature not in ('label_ids', 'label')]
        dataset = dataset.remove_columns(unwanted_columns)
        output_signature, _ = dataset._get_output_signature(dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args)
    output_columns = list(output_signature.keys())
    feature_cols = [col for col in output_columns if col in model_inputs and col not in model_labels]
    label_cols = [col for col in output_columns if col in model_labels]
    feature_cols = feature_cols[0] if len(feature_cols) == 1 else feature_cols
    label_cols = label_cols[0] if len(label_cols) == 1 else label_cols
    if drop_remainder is None:
        drop_remainder = shuffle
    tf_dataset = dataset.to_tf_dataset(columns=feature_cols, label_cols=label_cols, batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder, collate_fn=collate_fn, collate_fn_args=collate_fn_args, prefetch=prefetch)
    return tf_dataset