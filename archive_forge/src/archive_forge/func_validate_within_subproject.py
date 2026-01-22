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
def validate_within_subproject(self, subdir, fname):
    srcdir = Path(self.environment.source_dir)
    builddir = Path(self.environment.build_dir)
    if isinstance(fname, P_OBJ.DependencyVariableString):

        def validate_installable_file(fpath: Path) -> bool:
            installablefiles: T.Set[Path] = set()
            for d in self.build.data:
                for s in d.sources:
                    installablefiles.add(Path(s.absolute_path(srcdir, builddir)))
            installabledirs = [str(Path(srcdir, s.source_subdir)) for s in self.build.install_dirs]
            if fpath in installablefiles:
                return True
            for d in installabledirs:
                if str(fpath).startswith(d):
                    return True
            return False
        norm = Path(fname)
        if validate_installable_file(norm):
            return
    norm = Path(os.path.abspath(Path(srcdir, subdir, fname)))
    if os.path.isdir(norm):
        inputtype = 'directory'
    else:
        inputtype = 'file'
    if InterpreterRuleRelaxation.ALLOW_BUILD_DIR_FILE_REFERENCES in self.relaxations and builddir in norm.parents:
        return
    if srcdir not in norm.parents:
        return
    project_root = Path(srcdir, self.root_subdir)
    subproject_dir = project_root / self.subproject_dir
    if norm == project_root:
        return
    if project_root not in norm.parents:
        raise InterpreterException(f'Sandbox violation: Tried to grab {inputtype} {norm.name} outside current (sub)project.')
    if subproject_dir == norm or subproject_dir in norm.parents:
        raise InterpreterException(f'Sandbox violation: Tried to grab {inputtype} {norm.name} from a nested subproject.')