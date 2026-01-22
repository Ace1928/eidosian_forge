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
@noPosargs
@typed_kwargs('vcs_tag', CT_INPUT_KW.evolve(required=True), MULTI_OUTPUT_KW, KwargInfo('command', ContainerTypeInfo(list, (str, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, ExternalProgram, mesonlib.File)), listify=True, default=[]), KwargInfo('fallback', (str, NoneType)), KwargInfo('replace_string', str, default='@VCS_TAG@'))
def func_vcs_tag(self, node: mparser.BaseNode, args: T.List['TYPE_var'], kwargs: 'kwtypes.VcsTag') -> build.CustomTarget:
    if kwargs['fallback'] is None:
        FeatureNew.single_use('Optional fallback in vcs_tag', '0.41.0', self.subproject, location=node)
    fallback = kwargs['fallback'] or self.project_version
    replace_string = kwargs['replace_string']
    regex_selector = '(.*)'
    vcs_cmd = kwargs['command']
    source_dir = os.path.normpath(os.path.join(self.environment.get_source_dir(), self.subdir))
    if vcs_cmd:
        if isinstance(vcs_cmd[0], (str, mesonlib.File)):
            if isinstance(vcs_cmd[0], mesonlib.File):
                FeatureNew.single_use('vcs_tag with file as the first argument', '0.62.0', self.subproject, location=node)
            maincmd = self.find_program_impl(vcs_cmd[0], required=False)
            if maincmd.found():
                vcs_cmd[0] = maincmd
        else:
            FeatureNew.single_use('vcs_tag with custom_tgt, external_program, or exe as the first argument', '0.63.0', self.subproject, location=node)
    else:
        vcs = mesonlib.detect_vcs(source_dir)
        if vcs:
            mlog.log('Found {} repository at {}'.format(vcs['name'], vcs['wc_dir']))
            vcs_cmd = vcs['get_rev'].split()
            regex_selector = vcs['rev_regex']
        else:
            vcs_cmd = [' ']
    self._validate_custom_target_outputs(len(kwargs['input']) > 1, kwargs['output'], 'vcs_tag')
    cmd = self.environment.get_build_command() + ['--internal', 'vcstagger', '@INPUT0@', '@OUTPUT0@', fallback, source_dir, replace_string, regex_selector] + vcs_cmd
    tg = build.CustomTarget(kwargs['output'][0], self.subdir, self.subproject, self.environment, cmd, self.source_strings_to_files(kwargs['input']), kwargs['output'], build_by_default=True, build_always_stale=True)
    self.add_target(tg.name, tg)
    return tg