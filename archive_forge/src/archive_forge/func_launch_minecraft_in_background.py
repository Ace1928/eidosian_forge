from __future__ import print_function
from builtins import str
from builtins import range
import io
import os
import platform
import socket
import subprocess
import sys
import time
def launch_minecraft_in_background(minecraft_path, ports=None, timeout=360, replaceable=False, score=False, max_mem=None):
    if ports is None:
        ports = []
    if len(ports) == 0:
        ports = [10000]
    processes = []
    for port in ports:
        if _port_has_listener(port):
            print('Something is listening on port', port, '- will assume Minecraft is running.')
            continue
        replaceable_arg = ' -replaceable ' if replaceable else ''
        scorepolicy_arg = ' -scorepolicy ' if score else ''
        scorepolicy_value = ' 2 ' if score else ''
        maxmem_arg = ' -maxMem ' if max_mem is not None else ''
        maxmem_value = ' ' + max_mem + ' ' if max_mem else ''
        print('Nothing is listening on port', port, '- will attempt to launch Minecraft from a new terminal.')
        if os.name == 'nt':
            args = [minecraft_path + '/launchClient.bat', '-port', str(port), replaceable_arg.strip(), scorepolicy_arg.strip(), scorepolicy_value.strip(), maxmem_arg.strip(), maxmem_value.strip()]
            p = subprocess.Popen([arg for arg in args if arg != ''], creationflags=subprocess.CREATE_NEW_CONSOLE, close_fds=True)
        elif sys.platform == 'darwin':
            launcher_file = '/tmp/launcher_' + str(os.getpid()) + '.sh'
            tmp_file = open(launcher_file, 'w')
            tmp_file.write(minecraft_path + '/launchClient.sh -port ' + str(port) + replaceable_arg + scorepolicy_arg + scorepolicy_value + maxmem_arg + maxmem_value)
            tmp_file.close()
            os.chmod(launcher_file, 448)
            p = subprocess.Popen(['open', '-a', 'Terminal.app', launcher_file])
        else:
            p = subprocess.Popen(minecraft_path + '/launchClient.sh -port ' + str(port) + replaceable_arg + scorepolicy_arg + scorepolicy_value + maxmem_arg + maxmem_value, close_fds=True, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)
        print('Giving Minecraft some time to launch... ')
        launched = False
        for _ in range(timeout // 3):
            print('.', end=' ')
            time.sleep(3)
            if _port_has_listener(port):
                print('ok')
                launched = True
                break
        if not launched:
            print('Minecraft not yet launched. Giving up.')
            exit(1)
    return processes