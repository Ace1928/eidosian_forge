import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def monkey_patch() -> ContextManager:
    """
    Monkey-patch the system argparse module to automatically page help output.

    The result of calling this function can optionally be used as a context
    manager to restore the status quo when it exits.
    """
    import sys

    def get_existing_classes(module: types.ModuleType) -> Tuple[Type, ...]:
        return (module._HelpAction, module.HelpFormatter, module.RawDescriptionHelpFormatter, module.RawTextHelpFormatter, module.ArgumentDefaultsHelpFormatter, module.MetavarTypeHelpFormatter)

    def patch_classes(module: types.ModuleType, impl: Tuple[Type, ...]) -> None:
        module._HelpAction, module.HelpFormatter, module.RawDescriptionHelpFormatter, module.RawTextHelpFormatter, module.ArgumentDefaultsHelpFormatter, module.MetavarTypeHelpFormatter = impl
    orig = get_existing_classes(argparse)
    orig_fmtr = argparse.ArgumentParser._get_formatter
    patched = get_existing_classes(sys.modules[__name__])
    patch_classes(argparse, patched)
    new_fmtr = _substitute_formatter(orig_fmtr)
    argparse.ArgumentParser._get_formatter = new_fmtr

    @contextlib.contextmanager
    def unpatcher() -> Generator:
        try:
            yield
        finally:
            patch_classes(argparse, orig)
            argparse.ArgumentParser._get_formatter = orig_fmtr
    return unpatcher()