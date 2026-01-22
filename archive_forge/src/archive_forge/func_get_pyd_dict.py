import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
def get_pyd_dict(model: typing.Union[BaseModel, BaseSettings], **kwargs) -> typing.Dict[str, typing.Any]:
    """
    Get a dict from a pydantic model
    """
    if kwargs:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return model.model_dump(**kwargs) if PYD_VERSION == 2 else model.dict(**kwargs)