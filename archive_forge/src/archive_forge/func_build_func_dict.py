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
def build_func_dict(self) -> None:
    self.funcs.update({'add_global_arguments': self.func_add_global_arguments, 'add_global_link_arguments': self.func_add_global_link_arguments, 'add_languages': self.func_add_languages, 'add_project_arguments': self.func_add_project_arguments, 'add_project_dependencies': self.func_add_project_dependencies, 'add_project_link_arguments': self.func_add_project_link_arguments, 'add_test_setup': self.func_add_test_setup, 'alias_target': self.func_alias_target, 'assert': self.func_assert, 'benchmark': self.func_benchmark, 'both_libraries': self.func_both_lib, 'build_target': self.func_build_target, 'configuration_data': self.func_configuration_data, 'configure_file': self.func_configure_file, 'custom_target': self.func_custom_target, 'debug': self.func_debug, 'declare_dependency': self.func_declare_dependency, 'dependency': self.func_dependency, 'disabler': self.func_disabler, 'environment': self.func_environment, 'error': self.func_error, 'executable': self.func_executable, 'files': self.func_files, 'find_program': self.func_find_program, 'generator': self.func_generator, 'get_option': self.func_get_option, 'get_variable': self.func_get_variable, 'import': self.func_import, 'include_directories': self.func_include_directories, 'install_data': self.func_install_data, 'install_emptydir': self.func_install_emptydir, 'install_headers': self.func_install_headers, 'install_man': self.func_install_man, 'install_subdir': self.func_install_subdir, 'install_symlink': self.func_install_symlink, 'is_disabler': self.func_is_disabler, 'is_variable': self.func_is_variable, 'jar': self.func_jar, 'join_paths': self.func_join_paths, 'library': self.func_library, 'message': self.func_message, 'option': self.func_option, 'project': self.func_project, 'range': self.func_range, 'run_command': self.func_run_command, 'run_target': self.func_run_target, 'set_variable': self.func_set_variable, 'structured_sources': self.func_structured_sources, 'subdir': self.func_subdir, 'shared_library': self.func_shared_lib, 'shared_module': self.func_shared_module, 'static_library': self.func_static_lib, 'subdir_done': self.func_subdir_done, 'subproject': self.func_subproject, 'summary': self.func_summary, 'test': self.func_test, 'unset_variable': self.func_unset_variable, 'vcs_tag': self.func_vcs_tag, 'warning': self.func_warning})
    if 'MESON_UNIT_TEST' in os.environ:
        self.funcs.update({'exception': self.func_exception})
    if 'MESON_RUNNING_IN_PROJECT_TESTS' in os.environ:
        self.funcs.update({'expect_error': self.func_expect_error})