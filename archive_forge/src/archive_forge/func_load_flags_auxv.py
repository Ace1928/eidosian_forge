import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def load_flags_auxv(self):
    auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV='1'))
    for at in auxv.split(b'\n'):
        if not at.startswith(b'AT_HWCAP'):
            continue
        hwcap_value = [s.strip() for s in at.split(b':', 1)]
        if len(hwcap_value) == 2:
            self.features_flags = self.features_flags.union(hwcap_value[1].upper().decode().split())