from __future__ import annotations
import os
import shutil
import typing as T
import xml.etree.ElementTree as ET
from . import ModuleReturnValue, ExtensionModule
from .. import build
from .. import coredata
from .. import mlog
from ..dependencies import find_external_dependency, Dependency, ExternalLibrary, InternalDependency
from ..mesonlib import MesonException, File, version_compare, Popen_safe
from ..interpreter import extract_required_kwarg
from ..interpreter.type_checking import INSTALL_DIR_KW, INSTALL_KW, NoneType
from ..interpreterbase import ContainerTypeInfo, FeatureDeprecated, KwargInfo, noPosargs, FeatureNew, typed_kwargs
from ..programs import NonExistingExternalProgram
def gen_bins() -> T.Generator[T.Tuple[str, str], None, None]:
    for b in self.tools:
        if qt_dep.bindir:
            yield (os.path.join(qt_dep.bindir, b), b)
        if qt_dep.libexecdir:
            yield (os.path.join(qt_dep.libexecdir, b), b)
        yield (f'{b}{qt_dep.qtver}', b)
        yield (f'{b}-qt{qt_dep.qtver}', b)
        yield (b, b)