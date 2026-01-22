from __future__ import annotations as _annotations
import re
from typing_extensions import Literal, Self
from ._migration import getattr_migration
from .version import version_short
@classmethod
def from_name_error(cls, name_error: NameError) -> Self:
    """Convert a `NameError` to a `PydanticUndefinedAnnotation` error.

        Args:
            name_error: `NameError` to be converted.

        Returns:
            Converted `PydanticUndefinedAnnotation` error.
        """
    try:
        name = name_error.name
    except AttributeError:
        name = re.search(".*'(.+?)'", str(name_error)).group(1)
    return cls(name=name, message=str(name_error))