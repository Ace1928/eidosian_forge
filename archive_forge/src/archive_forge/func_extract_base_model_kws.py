from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def extract_base_model_kws(model: MT, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extracts the kwargs from the resource and returns the kwargs and the model kwargs
    """
    global _extracted_base_model_kws
    base_model_name = f'{model.__module__}.{model.__name__}'
    if base_model_name not in _extracted_base_model_kws:
        from lazyops.types.models import get_pyd_field_names
        resource_kws = get_pyd_field_names(model)
        _extracted_base_model_kws[base_model_name] = resource_kws
    model_kwargs = {key: kwargs.pop(key) for key in kwargs if key in _extracted_base_model_kws[base_model_name]}
    return (kwargs, model_kwargs)