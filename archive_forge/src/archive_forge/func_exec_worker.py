import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional
from ray.util.annotations import DeveloperAPI
from ray.core.generated.common_pb2 import Language
from ray._private.services import get_ray_jars_dir
from ray._private.utils import update_envs
def exec_worker(self, passthrough_args: List[str], language: Language):
    update_envs(self.env_vars)
    if language == Language.PYTHON and sys.platform == 'win32':
        executable = self.py_executable
    elif language == Language.PYTHON:
        executable = f'exec {self.py_executable}'
    elif language == Language.JAVA:
        executable = 'java'
        ray_jars = os.path.join(get_ray_jars_dir(), '*')
        local_java_jars = []
        for java_jar in self.java_jars:
            local_java_jars.append(f'{java_jar}/*')
            local_java_jars.append(java_jar)
        class_path_args = ['-cp', ray_jars + ':' + str(':'.join(local_java_jars))]
        passthrough_args = class_path_args + passthrough_args
    elif sys.platform == 'win32':
        executable = ''
    else:
        executable = 'exec '
    default_worker_path = self.container.get('worker_path')
    if self.container and default_worker_path:
        logger.debug(f'Changing the default worker path from {passthrough_args[0]} to {default_worker_path}.')
        passthrough_args[0] = default_worker_path
    passthrough_args = [s.replace(' ', '\\ ') for s in passthrough_args]
    exec_command = ' '.join([f'{executable}'] + passthrough_args)
    command_str = ' '.join(self.command_prefix + [exec_command])
    MACOS_LIBRARY_PATH_ENV_NAME = 'DYLD_LIBRARY_PATH'
    if MACOS_LIBRARY_PATH_ENV_NAME in os.environ:
        command_str = MACOS_LIBRARY_PATH_ENV_NAME + '=' + os.environ.get(MACOS_LIBRARY_PATH_ENV_NAME) + ' ' + command_str
    logger.debug(f"Exec'ing worker with command: {command_str}")
    if sys.platform == 'win32':
        cmd = [*self.command_prefix, executable, *passthrough_args]
        subprocess.Popen(cmd, shell=True).wait()
    else:
        os.execvp('bash', args=['bash', '-c', command_str])