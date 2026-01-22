from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def deprecatedModuleAttribute(version, message, moduleName, name):
    """
    Declare a module-level attribute as being deprecated.

    @type version: L{incremental.Version}
    @param version: Version that the attribute was deprecated in

    @type message: C{str}
    @param message: Deprecation message

    @type moduleName: C{str}
    @param moduleName: Fully-qualified Python name of the module containing
        the deprecated attribute; if called from the same module as the
        attributes are being deprecated in, using the C{__name__} global can
        be helpful

    @type name: C{str}
    @param name: Attribute name to deprecate
    """
    module = sys.modules[moduleName]
    if not isinstance(module, _ModuleProxy):
        module = cast(ModuleType, _ModuleProxy(module))
        sys.modules[moduleName] = module
    _deprecateAttribute(module, name, version, message)