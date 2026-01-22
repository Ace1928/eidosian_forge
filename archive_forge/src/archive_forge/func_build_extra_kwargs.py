import contextlib
import datetime
import functools
import importlib
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from packaging.version import parse
from requests import HTTPError, Response
from langchain_core.pydantic_v1 import SecretStr
def build_extra_kwargs(extra_kwargs: Dict[str, Any], values: Dict[str, Any], all_required_field_names: Set[str]) -> Dict[str, Any]:
    """Build extra kwargs from values and extra_kwargs.

    Args:
        extra_kwargs: Extra kwargs passed in by user.
        values: Values passed in by user.
        all_required_field_names: All required field names for the pydantic class.
    """
    for field_name in list(values):
        if field_name in extra_kwargs:
            raise ValueError(f'Found {field_name} supplied twice.')
        if field_name not in all_required_field_names:
            warnings.warn(f'WARNING! {field_name} is not default parameter.\n                {field_name} was transferred to model_kwargs.\n                Please confirm that {field_name} is what you intended.')
            extra_kwargs[field_name] = values.pop(field_name)
    invalid_model_kwargs = all_required_field_names.intersection(extra_kwargs.keys())
    if invalid_model_kwargs:
        raise ValueError(f'Parameters {invalid_model_kwargs} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.')
    return extra_kwargs