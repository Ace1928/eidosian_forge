from __future__ import annotations
import itertools, os, re
import typing as T
from .. import compilers
from ..build import (CustomTarget, BuildTarget,
from ..coredata import UserFeatureOption
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase.decorators import KwargInfo, ContainerTypeInfo
from ..mesonlib import (File, FileMode, MachineChoice, listify, has_path_sep,
from ..programs import ExternalProgram
def link_whole_validator(values: T.List[T.Union[StaticLibrary, CustomTarget, CustomTargetIndex, Dependency]]) -> T.Optional[str]:
    for l in values:
        if isinstance(l, (CustomTarget, CustomTargetIndex)) and l.links_dynamically():
            return f'{type(l).__name__} returning a shared library is not allowed'
        if isinstance(l, Dependency):
            return _link_with_error
    return None