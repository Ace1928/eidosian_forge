from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
@typed_pos_args('generator.process', min_varargs=1, varargs=(str, mesonlib.File, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList))
@typed_kwargs('generator.process', KwargInfo('preserve_path_from', (str, NoneType), since='0.45.0'), KwargInfo('extra_args', ContainerTypeInfo(list, str), listify=True, default=[]), ENV_KW.evolve(since='1.3.0'))
def process_method(self, args: T.Tuple[T.List[T.Union[str, mesonlib.File, 'build.GeneratedTypes']]], kwargs: 'kwargs.GeneratorProcess') -> build.GeneratedList:
    preserve_path_from = kwargs['preserve_path_from']
    if preserve_path_from is not None:
        preserve_path_from = os.path.normpath(preserve_path_from)
        if not os.path.isabs(preserve_path_from):
            raise InvalidArguments('Preserve_path_from must be an absolute path for now. Sorry.')
    if any((isinstance(a, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)) for a in args[0])):
        FeatureNew.single_use('Calling generator.process with CustomTarget or Index of CustomTarget.', '0.57.0', self.interpreter.subproject)
    gl = self.held_object.process_files(args[0], self.interpreter, preserve_path_from, extra_args=kwargs['extra_args'], env=kwargs['env'])
    return gl