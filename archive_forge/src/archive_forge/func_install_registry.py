from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def install_registry(self, registry):
    """
        Install a *registry* (a templates.Registry instance) of function,
        attribute and global declarations.
        """
    try:
        loader = self._registries[registry]
    except KeyError:
        loader = templates.RegistryLoader(registry)
        self._registries[registry] = loader
    for ftcls in loader.new_registrations('functions'):
        self.insert_function(ftcls(self))
    for ftcls in loader.new_registrations('attributes'):
        self.insert_attributes(ftcls(self))
    for gv, gty in loader.new_registrations('globals'):
        existing = self._lookup_global(gv)
        if existing is None:
            self.insert_global(gv, gty)
        else:
            newty = existing.augment(gty)
            if newty is None:
                raise TypeError('cannot augment %s with %s' % (existing, gty))
            self._remove_global(gv)
            self._insert_global(gv, newty)