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
@typed_pos_args('fs.is_samepath', (str, File), (str, File))
def is_samepath(self, state: 'ModuleState', args: T.Tuple['FileOrString', 'FileOrString'], kwargs: T.Dict[str, T.Any]) -> bool:
    if isinstance(args[0], File) or isinstance(args[1], File):
        FeatureNew('fs.is_samepath with file', '0.59.0').use(state.subproject, location=state.current_node)
    file1 = self._resolve_dir(state, args[0])
    file2 = self._resolve_dir(state, args[1])
    if not file1.exists():
        return False
    if not file2.exists():
        return False
    try:
        return file1.samefile(file2)
    except OSError:
        return False