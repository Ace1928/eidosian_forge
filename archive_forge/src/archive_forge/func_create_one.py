import os
import pathlib
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Generic, TYPE_CHECKING
from lazyops.types.formatting import to_camel_case, to_snake_case, to_graphql_format
from lazyops.types.classprops import classproperty, lazyproperty
from lazyops.utils.serialization import Json
from pydantic import Field
from pydantic.networks import AnyUrl
from lazyops.imports._pydantic import BaseSettings as _BaseSettings
from lazyops.imports._pydantic import BaseModel as _BaseModel
from lazyops.imports._pydantic import (
@classmethod
def create_one(cls, **kwargs) -> Tuple[Type['BaseModel'], Dict]:
    """
        Extracts the resource from the kwargs and returns the resource 
        and the remaining kwargs
        """
    resource_fields = get_pyd_field_names(cls)
    resource_kwargs = {k: v for k, v in kwargs.items() if k in resource_fields}
    return_kwargs = {k: v for k, v in kwargs.items() if k not in resource_fields}
    resource_obj = pyd_parse_obj(cls, resource_kwargs)
    return (resource_obj, return_kwargs)