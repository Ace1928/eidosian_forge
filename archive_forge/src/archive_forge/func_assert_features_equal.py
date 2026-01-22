import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def assert_features_equal(actual, desired, fname):
    __tracebackhide__ = True
    actual, desired = (str(actual), str(desired))
    if actual == desired:
        return
    detected = str(__cpu_features__).replace("'", '')
    try:
        with open('/proc/cpuinfo') as fd:
            cpuinfo = fd.read(2048)
    except Exception as err:
        cpuinfo = str(err)
    try:
        import subprocess
        auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV='1'))
        auxv = auxv.decode()
    except Exception as err:
        auxv = str(err)
    import textwrap
    error_report = textwrap.indent('\n###########################################\n### Extra debugging information\n###########################################\n-------------------------------------------\n--- NumPy Detections\n-------------------------------------------\n%s\n-------------------------------------------\n--- SYS / CPUINFO\n-------------------------------------------\n%s....\n-------------------------------------------\n--- SYS / AUXV\n-------------------------------------------\n%s\n' % (detected, cpuinfo, auxv), prefix='\r')
    raise AssertionError("Failure Detection\n NAME: '%s'\n ACTUAL: %s\n DESIRED: %s\n%s" % (fname, actual, desired, error_report))