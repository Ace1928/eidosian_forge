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
@FeatureNew('add_project_dependencies', '0.63.0')
@typed_pos_args('add_project_dependencies', varargs=dependencies.Dependency)
@typed_kwargs('add_project_dependencies', NATIVE_KW, LANGUAGE_KW)
def func_add_project_dependencies(self, node: mparser.FunctionNode, args: T.Tuple[T.List[dependencies.Dependency]], kwargs: 'kwtypes.FuncAddProjectArgs') -> None:
    for_machine = kwargs['native']
    for lang in kwargs['language']:
        if lang not in self.compilers[for_machine]:
            raise InvalidCode(f'add_project_dependencies() called before add_language() for language "{lang}"')
    for d in dependencies.get_leaf_external_dependencies(args[0]):
        compile_args = list(d.get_compile_args())
        system_incdir = d.get_include_type() == 'system'
        for i in d.get_include_dirs():
            for lang in kwargs['language']:
                comp = self.coredata.compilers[for_machine][lang]
                for idir in i.to_string_list(self.environment.get_source_dir(), self.environment.get_build_dir()):
                    compile_args.extend(comp.get_include_args(idir, system_incdir))
        self._add_project_arguments(node, self.build.projects_args[for_machine], compile_args, kwargs)
        self._add_project_arguments(node, self.build.projects_link_args[for_machine], d.get_link_args(), kwargs)