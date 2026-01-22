import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
def get_pyd_fields_dict(model: typing.Type[typing.Union[BaseModel, BaseSettings]]) -> typing.Dict[str, FieldInfo]:
    """
    Get a dict of fields from a pydantic model
    """
    return model.model_fields if PYD_VERSION == 2 else model.__fields__