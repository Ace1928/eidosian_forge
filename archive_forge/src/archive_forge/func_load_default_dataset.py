import copy
import functools
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from datasets import Dataset, DatasetDict
from datasets import load_dataset as datasets_load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from .. import logging
def load_default_dataset(self, only_keep_necessary_columns: bool=False, load_smallest_split: bool=False, num_samples: Optional[int]=None, shuffle: bool=False, **load_dataset_kwargs):
    if isinstance(self.DEFAULT_DATASET_ARGS, dict):
        path = self.DEFAULT_DATASET_ARGS.get('path', None)
        if path is None:
            raise ValueError('When DEFAULT_DATASET_ARGS is a dictionary, it must contain a key called "path" corresponding to the path or name of the dataset.')
        common_keys = set(self.DEFAULT_DATASET_ARGS.keys()) & set(load_dataset_kwargs.keys())
        if common_keys:
            ', '.join(common_keys)
            logger.warning('The following provided arguments will be overriden because they are hardcoded when using load_default_dataset: {override_config_key}.')
        kwargs = copy.deepcopy(load_dataset_kwargs)
        kwargs.update({k: v for k, v in self.DEFAULT_DATASET_ARGS.items() if k != 'path'})
    else:
        path = self.DEFAULT_DATASET_ARGS
        kwargs = load_dataset_kwargs
    return self.load_dataset(path, data_keys=self.DEFAUL_DATASET_DATA_KEYS, ref_keys=self.DEFAULT_REF_KEYS, only_keep_necessary_columns=only_keep_necessary_columns, load_smallest_split=load_smallest_split, num_samples=num_samples, shuffle=shuffle, **kwargs)