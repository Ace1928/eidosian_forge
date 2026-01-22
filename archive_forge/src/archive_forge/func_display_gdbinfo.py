from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
def display_gdbinfo(sep_pos=45):
    """Displays the information collected by collect_gdbinfo.
    """
    gdb_info = collect_gdbinfo()
    print('-' * 80)
    fmt = f'%-{sep_pos}s : %-s'
    print(fmt % ('Binary location', gdb_info.binary_loc))
    print(fmt % ('Print extension location', gdb_info.extension_loc))
    print(fmt % ('Python version', gdb_info.py_ver))
    print(fmt % ('NumPy version', gdb_info.np_ver))
    print(fmt % ('Numba printing extension support', gdb_info.supported))
    print('')
    print('To load the Numba gdb printing extension, execute the following from the gdb prompt:')
    print(f'\nsource {gdb_info.extension_loc}\n')
    print('-' * 80)
    warn = '\n    =============================================================\n    IMPORTANT: Before sharing you should remove any information\n    in the above that you wish to keep private e.g. paths.\n    =============================================================\n    '
    print(dedent(warn))