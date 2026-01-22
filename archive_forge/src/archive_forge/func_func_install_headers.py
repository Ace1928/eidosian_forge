from __future__ import annotations
from .. import mparser
from .. import environment
from .. import coredata
from .. import dependencies
from .. import mlog
from .. import build
from .. import optinterpreter
from .. import compilers
from .. import envconfig
from ..wrap import wrap, WrapMode
from .. import mesonlib
from ..mesonlib import (EnvironmentVariables, ExecutableSerialisation, MesonBugException, MesonException, HoldableObject,
from ..programs import ExternalProgram, NonExistingExternalProgram
from ..dependencies import Dependency
from ..depfile import DepFile
from ..interpreterbase import ContainerTypeInfo, InterpreterBase, KwargInfo, typed_kwargs, typed_pos_args
from ..interpreterbase import noPosargs, noKwargs, permittedKwargs, noArgsFlattening, noSecondLevelHolderResolving, unholder_return
from ..interpreterbase import InterpreterException, InvalidArguments, InvalidCode, SubdirDoneRequest
from ..interpreterbase import Disabler, disablerIfNotFound
from ..interpreterbase import FeatureNew, FeatureDeprecated, FeatureBroken, FeatureNewKwargs
from ..interpreterbase import ObjectHolder, ContextManagerObject
from ..interpreterbase import stringifyUserArguments
from ..modules import ExtensionModule, ModuleObject, MutableModuleObject, NewExtensionModule, NotFoundExtensionModule
from ..optinterpreter import optname_regex
from . import interpreterobjects as OBJ
from . import compiler as compilerOBJ
from .mesonmain import MesonMain
from .dependencyfallbacks import DependencyFallbacksHolder
from .interpreterobjects import (
from .type_checking import (
from . import primitives as P_OBJ
from pathlib import Path
from enum import Enum
import os
import shutil
import uuid
import re
import stat
import collections
import typing as T
import textwrap
import importlib
import copy
@typed_pos_args('install_headers', varargs=(str, mesonlib.File))
@typed_kwargs('install_headers', PRESERVE_PATH_KW, KwargInfo('subdir', (str, NoneType)), INSTALL_MODE_KW.evolve(since='0.47.0'), INSTALL_DIR_KW, INSTALL_FOLLOW_SYMLINKS)
def func_install_headers(self, node: mparser.BaseNode, args: T.Tuple[T.List['mesonlib.FileOrString']], kwargs: 'kwtypes.FuncInstallHeaders') -> build.Headers:
    install_mode = self._warn_kwarg_install_mode_sticky(kwargs['install_mode'])
    source_files = self.source_strings_to_files(args[0])
    install_subdir = kwargs['subdir']
    if install_subdir is not None:
        if kwargs['install_dir'] is not None:
            raise InterpreterException('install_headers: cannot specify both "install_dir" and "subdir". Use only "install_dir".')
        if os.path.isabs(install_subdir):
            mlog.deprecation('Subdir keyword must not be an absolute path. This will be a hard error in the next release.')
    else:
        install_subdir = ''
    dirs = collections.defaultdict(list)
    ret_headers = []
    if kwargs['preserve_path']:
        for file in source_files:
            dirname = os.path.dirname(file.fname)
            dirs[dirname].append(file)
    else:
        dirs[''].extend(source_files)
    for childdir in dirs:
        h = build.Headers(dirs[childdir], os.path.join(install_subdir, childdir), kwargs['install_dir'], install_mode, self.subproject, follow_symlinks=kwargs['follow_symlinks'])
        ret_headers.append(h)
        self.build.headers.append(h)
    return ret_headers