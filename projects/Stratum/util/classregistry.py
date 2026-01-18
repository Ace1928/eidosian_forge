"""
classregistry.py
================

This utility module provides a simple registry for introspecting and
manipulating the classes and functions defined within the `stratum`
package.  The goal of the registry is to allow both humans and
automated tools to discover, inspect and, if necessary, update
components of the simulation code in a structured way.  Over time
this facility can be extended to support more advanced features such
as dependency analysis, automated refactoring or code generation.

The registry works by using Python's built‑in introspection
capabilities via the ``inspect`` and ``importlib`` modules.  It
dynamically imports modules under the ``stratum`` namespace on
demand and inspects them for top‑level classes and functions.  The
registry does not import any modules on its own until asked to do
so, thus avoiding unnecessary import side effects.  Once a module
has been scanned it is cached for subsequent queries.

This initial implementation focuses on listing and retrieving source
code for classes and functions.  It intentionally does **not**
provide automatic editing or saving functionality because modifying
live Python modules at runtime can be error‑prone.  Instead, the
`get_source` method can be used to fetch the current source code for
an object, and external tooling (e.g. using ``container.apply_patch``)
may be used to update the source on disk.  Future work could
provide higher level helpers to perform safe in‑place edits.

Example usage::

    from stratum.util.classregistry import ClassRegistry
    reg = ClassRegistry()
    # List all modules under the stratum package
    print(reg.list_modules())
    # List classes in the quanta module
    print(reg.list_classes('stratum.core.quanta'))
    # Get source for a function
    src = reg.get_source('stratum.core.quanta', 'SignalQueue')
    print(src)

As the project evolves additional convenience APIs may be added.  See
the docstrings on individual methods for details.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from types import ModuleType
from typing import Dict, List, Optional, Tuple


class ClassRegistry:
    """A registry for discovering classes and functions in the Stratum codebase.

    This registry supports listing modules, classes and functions and
    retrieving the source code for a given object.  It does not
    perform any mutation of the underlying files but can be used in
    combination with external patching tools to implement a robust
    edit/update workflow.
    """

    def __init__(self, root_package: str = 'stratum') -> None:
        self.root_package = root_package
        # Cache of imported modules keyed by module name
        self._module_cache: Dict[str, ModuleType] = {}

    # ------------------------------------------------------------------
    # Module discovery

    def list_modules(self) -> List[str]:
        """Return a list of importable module names under the root package.

        The returned list includes both subpackages and modules found
        recursively under the root package directory.  Modules are
        returned using their fully qualified import paths (e.g.
        ``stratum.core.quanta``).  Modules are not imported until
        explicitly requested via ``load_module``.
        """
        modules: List[str] = []
        package = importlib.import_module(self.root_package)
        if not hasattr(package, '__path__'):
            return [self.root_package]
        for finder, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            modules.append(name)
        return sorted(modules)

    def load_module(self, module_name: str) -> ModuleType:
        """Import a module by name and cache it.

        Raises ImportError if the module cannot be imported.  Caches
        the result to avoid reimporting.  If the module has already
        been imported in this process (e.g. via other code), the
        existing module instance is returned.
        """
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        module = importlib.import_module(module_name)
        self._module_cache[module_name] = module
        return module

    # ------------------------------------------------------------------
    # Class and function discovery

    def list_classes(self, module_name: str) -> List[str]:
        """List the names of top level classes defined in the given module.

        Only classes defined at module scope are returned.  Classes
        imported from elsewhere are ignored.  If the module cannot be
        imported, an empty list is returned.
        """
        try:
            module = self.load_module(module_name)
        except ImportError:
            return []
        classes: List[str] = []
        for name, obj in vars(module).items():
            # only include classes defined in this module
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                classes.append(name)
        return sorted(classes)

    def list_functions(self, module_name: str) -> List[str]:
        """List the names of top level functions defined in the given module.

        Only functions defined at module scope are returned.  Nested
        functions or callables defined in classes are not included.
        If the module cannot be imported, an empty list is returned.
        """
        try:
            module = self.load_module(module_name)
        except ImportError:
            return []
        funcs: List[str] = []
        for name, obj in vars(module).items():
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                funcs.append(name)
        return sorted(funcs)

    # ------------------------------------------------------------------
    # Source retrieval

    def get_source(self, module_name: str, obj_name: str) -> Optional[str]:
        """Return the source code for a class or function.

        If the object cannot be found or its source cannot be retrieved
        (e.g. builtins or C extensions), ``None`` is returned.  The
        returned string includes the decorators and signature if
        present.  Users should take care when editing code via
        external tools to preserve indentation and context.
        """
        try:
            module = self.load_module(module_name)
        except ImportError:
            return None
        obj = getattr(module, obj_name, None)
        if obj is None:
            return None
        try:
            src = inspect.getsource(obj)
        except (TypeError, OSError):
            return None
        return src


__all__ = ['ClassRegistry']