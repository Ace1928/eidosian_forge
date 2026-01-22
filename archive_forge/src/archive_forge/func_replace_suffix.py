from __future__ import annotations
from pathlib import Path, PurePath, PureWindowsPath
import hashlib
import os
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import mlog
from ..build import BuildTarget, CustomTarget, CustomTargetIndex, InvalidArguments
from ..interpreter.type_checking import INSTALL_KW, INSTALL_MODE_KW, INSTALL_TAG_KW, NoneType
from ..interpreterbase import FeatureNew, KwargInfo, typed_kwargs, typed_pos_args, noKwargs
from ..mesonlib import File, MesonException, has_path_sep, path_is_in_root, relpath
@noKwargs
@typed_pos_args('fs.replace_suffix', (str, File, CustomTarget, CustomTargetIndex, BuildTarget), str)
def replace_suffix(self, state: 'ModuleState', args: T.Tuple[T.Union[FileOrString, BuildTargetTypes], str], kwargs: T.Dict[str, T.Any]) -> str:
    path = self._obj_to_path('fs.replace_suffix', args[0], state)
    return str(path.with_suffix(args[1]))