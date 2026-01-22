import ast
import contextlib
import csv
import inspect
import logging
import os
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, Type, Union
from warnings import warn
import torch
import yaml
from lightning_utilities.core.apply_func import apply_to_collection
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.data import AttributeDict
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import parse_class_init_keys
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def save_hparams_to_yaml(config_yaml: _PATH, hparams: Union[dict, Namespace], use_omegaconf: bool=True) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
        use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
            the hparams will be converted to ``DictConfig`` if possible.

    """
    fs = get_filesystem(config_yaml)
    if not _is_dir(fs, os.path.dirname(config_yaml)):
        raise RuntimeError(f'Missing folder: {os.path.dirname(config_yaml)}.')
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif isinstance(hparams, AttributeDict):
        hparams = dict(hparams)
    if _OMEGACONF_AVAILABLE and use_omegaconf:
        from omegaconf import OmegaConf
        from omegaconf.dictconfig import DictConfig
        from omegaconf.errors import UnsupportedValueType, ValidationError
        hparams = deepcopy(hparams)
        hparams = apply_to_collection(hparams, DictConfig, OmegaConf.to_container, resolve=True)
        with fs.open(config_yaml, 'w', encoding='utf-8') as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass
    if not isinstance(hparams, dict):
        raise TypeError('hparams must be dictionary')
    hparams_allowed = {}
    for k, v in hparams.items():
        try:
            v = v.name if isinstance(v, Enum) else v
            yaml.dump(v)
        except TypeError:
            warn(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v
    with fs.open(config_yaml, 'w', newline='') as fp:
        yaml.dump(hparams_allowed, fp)