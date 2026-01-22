import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_nvidia_smi():
    smi = 'nvidia-smi'
    if get_platform() == 'win32':
        system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
        program_files_root = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        legacy_path = os.path.join(program_files_root, 'NVIDIA Corporation', 'NVSMI', smi)
        new_path = os.path.join(system_root, 'System32', smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = '"{}"'.format(candidate_smi)
                break
    return smi