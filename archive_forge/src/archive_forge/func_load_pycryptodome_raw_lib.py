import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
def load_pycryptodome_raw_lib(name, cdecl):
    """Load a shared library and return a handle to it.

    @name,  the name of the library expressed as a PyCryptodome module,
            for instance Cryptodome.Cipher._raw_cbc.

    @cdecl, the C function declarations.
    """
    split = name.split('.')
    dir_comps, basename = (split[:-1], split[-1])
    attempts = []
    for ext in extension_suffixes:
        try:
            filename = basename + ext
            full_name = pycryptodome_filename(dir_comps, filename)
            if not os.path.isfile(full_name):
                attempts.append("Not found '%s'" % filename)
                continue
            return load_lib(full_name, cdecl)
        except OSError as exp:
            attempts.append("Cannot load '%s': %s" % (filename, str(exp)))
    raise OSError("Cannot load native module '%s': %s" % (name, ', '.join(attempts)))