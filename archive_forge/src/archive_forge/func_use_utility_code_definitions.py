from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def use_utility_code_definitions(scope, target, seen=None):
    if seen is None:
        seen = set()
    for entry in scope.entries.values():
        if entry in seen:
            continue
        seen.add(entry)
        if entry.used and entry.utility_code_definition:
            target.use_utility_code(entry.utility_code_definition)
            for required_utility in entry.utility_code_definition.requires:
                target.use_utility_code(required_utility)
        elif entry.as_module:
            use_utility_code_definitions(entry.as_module, target, seen)