from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
@typed_kwargs('gnome.post_install', KwargInfo('glib_compile_schemas', bool, default=False), KwargInfo('gio_querymodules', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('gtk_update_icon_cache', bool, default=False), KwargInfo('update_desktop_database', bool, default=False, since='0.59.0'), KwargInfo('update_mime_database', bool, default=False, since='0.64.0'))
@noPosargs
@FeatureNew('gnome.post_install', '0.57.0')
def post_install(self, state: 'ModuleState', args: T.List['TYPE_var'], kwargs: 'PostInstall') -> ModuleReturnValue:
    rv: T.List['mesonlib.ExecutableSerialisation'] = []
    datadir_abs = os.path.join(state.environment.get_prefix(), state.environment.get_datadir())
    if kwargs['glib_compile_schemas'] and (not self.install_glib_compile_schemas):
        self.install_glib_compile_schemas = True
        prog = self._find_tool(state, 'glib-compile-schemas')
        schemasdir = os.path.join(datadir_abs, 'glib-2.0', 'schemas')
        script = state.backend.get_executable_serialisation([prog, schemasdir])
        script.skip_if_destdir = True
        rv.append(script)
    for d in kwargs['gio_querymodules']:
        if d not in self.install_gio_querymodules:
            self.install_gio_querymodules.append(d)
            prog = self._find_tool(state, 'gio-querymodules')
            moduledir = os.path.join(state.environment.get_prefix(), d)
            script = state.backend.get_executable_serialisation([prog, moduledir])
            script.skip_if_destdir = True
            rv.append(script)
    if kwargs['gtk_update_icon_cache'] and (not self.install_gtk_update_icon_cache):
        self.install_gtk_update_icon_cache = True
        prog = state.find_program('gtk4-update-icon-cache', required=False)
        found = isinstance(prog, Executable) or prog.found()
        if not found:
            prog = state.find_program('gtk-update-icon-cache')
        icondir = os.path.join(datadir_abs, 'icons', 'hicolor')
        script = state.backend.get_executable_serialisation([prog, '-q', '-t', '-f', icondir])
        script.skip_if_destdir = True
        rv.append(script)
    if kwargs['update_desktop_database'] and (not self.install_update_desktop_database):
        self.install_update_desktop_database = True
        prog = state.find_program('update-desktop-database')
        appdir = os.path.join(datadir_abs, 'applications')
        script = state.backend.get_executable_serialisation([prog, '-q', appdir])
        script.skip_if_destdir = True
        rv.append(script)
    if kwargs['update_mime_database'] and (not self.install_update_mime_database):
        self.install_update_mime_database = True
        prog = state.find_program('update-mime-database')
        appdir = os.path.join(datadir_abs, 'mime')
        script = state.backend.get_executable_serialisation([prog, appdir])
        script.skip_if_destdir = True
        rv.append(script)
    return ModuleReturnValue(None, rv)