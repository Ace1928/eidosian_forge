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
@FeatureNew('both_libraries', '0.46.0')
def build_both_libraries(self, node: mparser.BaseNode, args: T.Tuple[str, SourcesVarargsType], kwargs: kwtypes.Library) -> build.BothLibraries:
    shared_lib = self.build_target(node, args, kwargs, build.SharedLibrary)
    static_lib = self.build_target(node, args, kwargs, build.StaticLibrary)
    if self.backend.name == 'xcode':
        reuse_object_files = False
    elif shared_lib.uses_rust():
        reuse_object_files = False
    elif any((k.endswith(('static_args', 'shared_args')) and v for k, v in kwargs.items())):
        reuse_object_files = False
    else:
        reuse_object_files = static_lib.pic
    if reuse_object_files:
        static_lib.objects.append(build.ExtractedObjects(shared_lib, shared_lib.sources, shared_lib.generated, []))
        static_lib.sources = []
        static_lib.generated = []
        static_lib.compilers = {k: v for k, v in static_lib.compilers.items() if k in compilers.clink_langs}
    return build.BothLibraries(shared_lib, static_lib)