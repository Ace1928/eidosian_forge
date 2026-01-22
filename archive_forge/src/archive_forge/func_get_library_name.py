import os
import sys
def get_library_name():
    """
    Return the name of the llvmlite shared library file.
    """
    if os.name == 'posix':
        if sys.platform == 'darwin':
            return 'libllvmlite.dylib'
        else:
            return 'libllvmlite.so'
    else:
        assert os.name == 'nt'
        return 'llvmlite.dll'