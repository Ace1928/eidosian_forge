from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def decorate_callable(self, func):
    if hasattr(func, 'patchings'):
        func.patchings.append(self)
        return func

    @wraps(func)
    def patched(*args, **keywargs):
        extra_args = []
        entered_patchers = []
        exc_info = tuple()
        try:
            for patching in patched.patchings:
                arg = patching.__enter__()
                entered_patchers.append(patching)
                if patching.attribute_name is not None:
                    keywargs.update(arg)
                elif patching.new is DEFAULT:
                    extra_args.append(arg)
            args += tuple(extra_args)
            return func(*args, **keywargs)
        except:
            if patching not in entered_patchers and _is_started(patching):
                entered_patchers.append(patching)
            exc_info = sys.exc_info()
            raise
        finally:
            for patching in reversed(entered_patchers):
                patching.__exit__(*exc_info)
    patched.patchings = [self]
    return patched