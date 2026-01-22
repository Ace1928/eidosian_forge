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
@typed_pos_args('install_data', varargs=(str, mesonlib.File))
@typed_kwargs('python_installation.install_sources', _PURE_KW, _SUBDIR_KW, PRESERVE_PATH_KW, KwargInfo('install_tag', (str, NoneType), since='0.60.0'))
def install_sources_method(self, args: T.Tuple[T.List[T.Union[str, mesonlib.File]]], kwargs: 'PyInstallKw') -> 'Data':
    self.held_object.run_bytecompile[self.version] = True
    tag = kwargs['install_tag'] or 'python-runtime'
    pure = kwargs['pure'] if kwargs['pure'] is not None else self.pure
    install_dir = self._get_install_dir_impl(pure, kwargs['subdir'])
    return self.interpreter.install_data_impl(self.interpreter.source_strings_to_files(args[0]), install_dir, mesonlib.FileMode(), rename=None, tag=tag, install_data_type='python', preserve_path=kwargs['preserve_path'])