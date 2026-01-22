from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform
def getGPUs():
    if platform.system() == 'Windows':
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = '%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe' % os.environ['systemdrive']
    else:
        nvidia_smi = 'nvidia-smi'
    try:
        p = Popen([nvidia_smi, '--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu', '--format=csv,noheader,nounits'], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')
    lines = output.split(os.linesep)
    numDevices = len(lines) - 1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        vals = line.split(', ')
        for i in range(12):
            if i == 0:
                deviceIds = int(vals[i])
            elif i == 1:
                uuid = vals[i]
            elif i == 2:
                gpuUtil = safeFloatCast(vals[i]) / 100
            elif i == 3:
                memTotal = safeFloatCast(vals[i])
            elif i == 4:
                memUsed = safeFloatCast(vals[i])
            elif i == 5:
                memFree = safeFloatCast(vals[i])
            elif i == 6:
                driver = vals[i]
            elif i == 7:
                gpu_name = vals[i]
            elif i == 8:
                serial = vals[i]
            elif i == 9:
                display_active = vals[i]
            elif i == 10:
                display_mode = vals[i]
            elif i == 11:
                temp_gpu = safeFloatCast(vals[i])
        GPUs.append(GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu))
    return GPUs