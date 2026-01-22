import os
import shutil
import subprocess
import tempfile
from ..printing import latex
from ..kinetics.rates import RadiolyticBase
from ..units import to_unitless, get_derived_unit
def ref_fmt(s):
    if s is None:
        return 'None'
    if tex:
        if isinstance(s, dict):
            return _doi(s['doi'])
        if s.startswith('doi:'):
            return _doi(s[4:])
    return s