import atexit
import errno
import importlib
import os
import sys
import tempfile
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
def load_source(source, delete_on_exit):
    """Loads the given source code as a Python module."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix='__autograph_generated_file', delete=False, encoding='utf-8') as f:
        module_name = os.path.basename(f.name[:-3])
        file_name = f.name
        f.write(source)
    if delete_on_exit:
        atexit.register(lambda: _remove_file(file_name))
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return (module, file_name)