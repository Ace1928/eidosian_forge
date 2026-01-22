import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_cpu_info(run_lambda):
    rc, out, err = (0, '', '')
    if get_platform() == 'linux':
        rc, out, err = run_lambda('lscpu')
    elif get_platform() == 'win32':
        rc, out, err = run_lambda('wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID,        CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE')
    elif get_platform() == 'darwin':
        rc, out, err = run_lambda('sysctl -n machdep.cpu.brand_string')
    cpu_info = 'None'
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info