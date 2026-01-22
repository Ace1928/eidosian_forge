from __future__ import annotations
import dataclasses
import typing as T
from .. import build, mesonlib
from ..build import IncludeDirs
from ..interpreterbase.decorators import noKwargs, noPosargs
from ..mesonlib import relpath, HoldableObject, MachineChoice
from ..programs import ExternalProgram
def process_include_dirs(self, dirs: T.Iterable[T.Union[str, IncludeDirs]]) -> T.Iterable[IncludeDirs]:
    """Convert raw include directory arguments to only IncludeDirs

        :param dirs: An iterable of strings and IncludeDirs
        :return: None
        :yield: IncludeDirs objects
        """
    for d in dirs:
        if isinstance(d, IncludeDirs):
            yield d
        else:
            yield self._interpreter.build_incdir_object([d])