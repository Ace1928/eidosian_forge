import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def emit_c_code(self, filename):
    from .recompiler import recompile
    if not hasattr(self, '_assigned_source'):
        raise ValueError('set_source() must be called before emit_c_code()')
    module_name, source, source_extension, kwds = self._assigned_source
    if source is None:
        raise TypeError('emit_c_code() is only for C extension modules, not for dlopen()-style pure Python modules')
    recompile(self, module_name, source, c_file=filename, call_c_compiler=False, **kwds)