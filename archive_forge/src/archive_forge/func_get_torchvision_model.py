import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Type, TypeVar
from lightning_utilities.core.imports import RequirementCache
from torch import nn
from typing_extensions import Concatenate, ParamSpec
import pytorch_lightning as pl
def get_torchvision_model(model_name: str, **kwargs: Any) -> nn.Module:
    from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
    if not _TORCHVISION_AVAILABLE:
        raise ModuleNotFoundError(str(_TORCHVISION_AVAILABLE))
    from torchvision import models
    torchvision_greater_equal_0_14 = RequirementCache('torchvision>=0.14.0')
    if torchvision_greater_equal_0_14:
        return models.get_model(model_name, **kwargs)
    return getattr(models, model_name)(**kwargs)