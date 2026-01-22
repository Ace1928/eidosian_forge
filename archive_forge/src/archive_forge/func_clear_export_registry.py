import re
import warnings
from numba.core import typing, sigutils
from numba.pycc.compiler import ExportEntry
def clear_export_registry():
    export_registry[:] = []