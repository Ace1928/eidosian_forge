import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
def get_pyd_fields(model: typing.Type[typing.Union[BaseModel, BaseSettings]]) -> typing.List[FieldInfo]:
    """
    Get a list of fields from a pydantic model
    """
    return list(get_pyd_fields_dict(model).values())