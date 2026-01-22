import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
@property
def module_version(self) -> str:
    """
        Returns the module version
        """
    return pkg_resources.get_distribution(self.module_name).version