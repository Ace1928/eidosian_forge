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
@noArgsFlattening
@typed_pos_args('environment', optargs=[(str, list, dict)])
@typed_kwargs('environment', ENV_METHOD_KW, ENV_SEPARATOR_KW.evolve(since='0.62.0'))
def func_environment(self, node: mparser.FunctionNode, args: T.Tuple[T.Union[None, str, T.List['TYPE_var'], T.Dict[str, 'TYPE_var']]], kwargs: 'TYPE_kwargs') -> EnvironmentVariables:
    init = args[0]
    if init is not None:
        FeatureNew.single_use('environment positional arguments', '0.52.0', self.subproject, location=node)
        msg = ENV_KW.validator(init)
        if msg:
            raise InvalidArguments(f'"environment": {msg}')
        if isinstance(init, dict) and any((i for i in init.values() if isinstance(i, list))):
            FeatureNew.single_use('List of string in dictionary value', '0.62.0', self.subproject, location=node)
        return env_convertor_with_method(init, kwargs['method'], kwargs['separator'])
    return EnvironmentVariables()