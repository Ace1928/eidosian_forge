from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def symbols_have_underscore_prefix(self, env: 'Environment') -> bool:
    """
        Check if the compiler prefixes an underscore to global C symbols.

        This overrides the Clike method, as for MSVC checking the
        underscore prefix based on the compiler define never works,
        so do not even try.
        """
    result = self._symbols_have_underscore_prefix_list(env)
    if result is not None:
        return result
    return self._symbols_have_underscore_prefix_searchbin(env)