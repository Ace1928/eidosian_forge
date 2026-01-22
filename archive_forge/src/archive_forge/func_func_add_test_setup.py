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
@typed_pos_args('add_test_setup', str)
@typed_kwargs('add_test_setup', KwargInfo('exe_wrapper', ContainerTypeInfo(list, (str, ExternalProgram)), listify=True, default=[]), KwargInfo('gdb', bool, default=False), KwargInfo('timeout_multiplier', int, default=1), KwargInfo('exclude_suites', ContainerTypeInfo(list, str), listify=True, default=[], since='0.57.0'), KwargInfo('is_default', bool, default=False, since='0.49.0'), ENV_KW)
def func_add_test_setup(self, node: mparser.BaseNode, args: T.Tuple[str], kwargs: 'kwtypes.AddTestSetup') -> None:
    setup_name = args[0]
    if re.fullmatch('([_a-zA-Z][_0-9a-zA-Z]*:)?[_a-zA-Z][_0-9a-zA-Z]*', setup_name) is None:
        raise InterpreterException('Setup name may only contain alphanumeric characters.')
    if ':' not in setup_name:
        setup_name = f'{(self.subproject if self.subproject else self.build.project_name)}:{setup_name}'
    exe_wrapper: T.List[str] = []
    for i in kwargs['exe_wrapper']:
        if isinstance(i, str):
            exe_wrapper.append(i)
        else:
            if not i.found():
                raise InterpreterException('Tried to use non-found executable.')
            exe_wrapper += i.get_command()
    timeout_multiplier = kwargs['timeout_multiplier']
    if timeout_multiplier <= 0:
        FeatureNew('add_test_setup() timeout_multiplier <= 0', '0.57.0').use(self.subproject)
    if kwargs['is_default']:
        if self.build.test_setup_default_name is not None:
            raise InterpreterException(f'{self.build.test_setup_default_name!r} is already set as default. is_default can be set to true only once')
        self.build.test_setup_default_name = setup_name
    self.build.test_setups[setup_name] = build.TestSetup(exe_wrapper, kwargs['gdb'], timeout_multiplier, kwargs['env'], kwargs['exclude_suites'])