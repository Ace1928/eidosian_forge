from __future__ import annotations
import copy, json, os, shutil, re
import typing as T
from . import ExtensionModule, ModuleInfo
from .. import mesonlib
from .. import mlog
from ..coredata import UserFeatureOption
from ..build import known_shmod_kwargs, CustomTarget, CustomTargetIndex, BuildTarget, GeneratedList, StructuredSources, ExtractedObjects, SharedModule
from ..dependencies import NotFoundDependency
from ..dependencies.detect import get_dep_identifier, find_external_dependency
from ..dependencies.python import BasicPythonExternalProgram, python_factory, _PythonDependencyBase
from ..interpreter import extract_required_kwarg, permitted_dependency_kwargs, primitives as P_OBJ
from ..interpreter.interpreterobjects import _ExternalProgramHolder
from ..interpreter.type_checking import NoneType, PRESERVE_PATH_KW, SHARED_MOD_KWS
from ..interpreterbase import (
from ..mesonlib import MachineChoice, OptionKey
from ..programs import ExternalProgram, NonExistingExternalProgram
@disablerIfNotFound
@typed_pos_args('python.find_installation', optargs=[str])
@typed_kwargs('python.find_installation', KwargInfo('required', (bool, UserFeatureOption), default=True), KwargInfo('disabler', bool, default=False, since='0.49.0'), KwargInfo('modules', ContainerTypeInfo(list, str), listify=True, default=[], since='0.51.0'), _PURE_KW.evolve(default=True, since='0.64.0'))
def find_installation(self, state: 'ModuleState', args: T.Tuple[T.Optional[str]], kwargs: 'FindInstallationKw') -> MaybePythonProg:
    feature_check = FeatureNew('Passing "feature" option to find_installation', '0.48.0')
    disabled, required, feature = extract_required_kwarg(kwargs, state.subproject, feature_check)
    np: T.List[str] = state.environment.lookup_binary_entry(MachineChoice.HOST, 'python') or []
    fallback = args[0]
    display_name = fallback or 'python'
    if not np and fallback is not None:
        np = [fallback]
    name_or_path = np[0] if np else None
    if disabled:
        mlog.log('Program', name_or_path or 'python', 'found:', mlog.red('NO'), '(disabled by:', mlog.bold(feature), ')')
        return NonExistingExternalProgram()
    python = self.installations.get(name_or_path)
    if not python:
        python = self._find_installation_impl(state, display_name, name_or_path, required)
        self.installations[name_or_path] = python
    want_modules = kwargs['modules']
    found_modules: T.List[str] = []
    missing_modules: T.List[str] = []
    if python.found() and want_modules:
        for mod in want_modules:
            p, *_ = mesonlib.Popen_safe(python.command + ['-c', f'import {mod}'])
            if p.returncode != 0:
                missing_modules.append(mod)
            else:
                found_modules.append(mod)
    msg: T.List['mlog.TV_Loggable'] = ['Program', python.name]
    if want_modules:
        msg.append('({})'.format(', '.join(want_modules)))
    msg.append('found:')
    if python.found() and (not missing_modules):
        msg.extend([mlog.green('YES'), '({})'.format(' '.join(python.command))])
    else:
        msg.append(mlog.red('NO'))
    if found_modules:
        msg.append('modules:')
        msg.append(', '.join(found_modules))
    mlog.log(*msg)
    if not python.found():
        if required:
            raise mesonlib.MesonException('{} not found'.format(name_or_path or 'python'))
        return NonExistingExternalProgram(python.name)
    elif missing_modules:
        if required:
            raise mesonlib.MesonException('{} is missing modules: {}'.format(name_or_path or 'python', ', '.join(missing_modules)))
        return NonExistingExternalProgram(python.name)
    else:
        assert isinstance(python, PythonExternalProgram), 'for mypy'
        python = copy.copy(python)
        python.pure = kwargs['pure']
        return python
    raise mesonlib.MesonBugException('Unreachable code was reached (PythonModule.find_installation).')