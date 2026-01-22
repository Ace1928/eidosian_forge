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
def source_strings_to_files(self, sources: T.List['SourceInputs'], strict: bool=True) -> T.List['SourceOutputs']:
    """Lower inputs to a list of Targets and Files, replacing any strings.

        :param sources: A raw (Meson DSL) list of inputs (targets, files, and
            strings)
        :raises InterpreterException: if any of the inputs are of an invalid type
        :return: A list of Targets and Files
        """
    mesonlib.check_direntry_issues(sources)
    if not isinstance(sources, list):
        sources = [sources]
    results: T.List['SourceOutputs'] = []
    for s in sources:
        if isinstance(s, str):
            if not strict and s.startswith(self.environment.get_build_dir()):
                results.append(s)
                mlog.warning(f'Source item {s!r} cannot be converted to File object, because it is a generated file. This will become a hard error in the future.', location=self.current_node)
            else:
                self.validate_within_subproject(self.subdir, s)
                results.append(mesonlib.File.from_source_file(self.environment.source_dir, self.subdir, s))
        elif isinstance(s, mesonlib.File):
            results.append(s)
        elif isinstance(s, (build.GeneratedList, build.BuildTarget, build.CustomTargetIndex, build.CustomTarget, build.ExtractedObjects, build.StructuredSources)):
            results.append(s)
        else:
            raise InterpreterException(f'Source item is {s!r} instead of string or File-type object')
    return results