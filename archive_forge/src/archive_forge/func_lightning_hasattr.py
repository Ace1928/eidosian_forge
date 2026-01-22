import copy
import inspect
import pickle
import types
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union
from torch import nn
import pytorch_lightning as pl
from lightning_fabric.utilities.data import AttributeDict as _AttributeDict
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def lightning_hasattr(model: 'pl.LightningModule', attribute: str) -> bool:
    """Special hasattr for Lightning.

    Checks for attribute in model namespace, the old hparams namespace/dict, and the datamodule.

    """
    return _lightning_get_first_attr_holder(model, attribute) is not None