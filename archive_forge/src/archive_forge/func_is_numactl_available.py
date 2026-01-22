import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
def is_numactl_available(self):
    numactl_available = False
    try:
        cmd = ['numactl', '-C', '0', '-m', '0', 'hostname']
        r = subprocess.run(cmd, env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if r.returncode == 0:
            numactl_available = True
    except Exception:
        pass
    return numactl_available