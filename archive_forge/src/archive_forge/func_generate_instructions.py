import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def generate_instructions(self):
    if getattr(self, 'error', False):
        return
    print(self.error)
    self.error = True
    if not self.cuda_available:
        self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA library was not detected or CUDA not installed.')
        self.add_log_entry('CUDA SETUP: Solution 1): Your paths are probably not up-to-date. You can update them via: sudo ldconfig.')
        self.add_log_entry('CUDA SETUP: Solution 2): If you do not have sudo rights, you can do the following:')
        self.add_log_entry('CUDA SETUP: Solution 2a): Find the cuda library via: find / -name libcuda.so 2>/dev/null')
        self.add_log_entry('CUDA SETUP: Solution 2b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_2a')
        self.add_log_entry('CUDA SETUP: Solution 2c): For a permanent solution add the export from 2b into your .bashrc file, located at ~/.bashrc')
        self.add_log_entry('CUDA SETUP: Solution 3): For a missing CUDA runtime library (libcudart.so), use `find / -name libcudart.so* and follow with step (2b)')
        return
    if self.cudart_path is None:
        self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.')
        self.add_log_entry('CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable')
        self.add_log_entry('CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null')
        self.add_log_entry('CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a')
        self.add_log_entry('CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc')
        self.add_log_entry('CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.')
        self.add_log_entry('CUDA SETUP: Solution 2a): Download CUDA install script: wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/cuda_install.sh')
        self.add_log_entry('CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.')
        self.add_log_entry('CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local')
        return
    make_cmd = f'CUDA_VERSION={self.cuda_version_string}'
    if len(self.cuda_version_string) < 3:
        make_cmd += ' make cuda92'
    elif self.cuda_version_string == '110':
        make_cmd += ' make cuda110'
    elif self.cuda_version_string[:2] == '11' and int(self.cuda_version_string[2]) > 0:
        make_cmd += ' make cuda11x'
    elif self.cuda_version_string[:2] == '12' and 1 >= int(self.cuda_version_string[2]) >= 0:
        make_cmd += ' make cuda12x'
    elif self.cuda_version_string == '100':
        self.add_log_entry('CUDA SETUP: CUDA 10.0 not supported. Please use a different CUDA version.')
        self.add_log_entry('CUDA SETUP: Before you try again running bitsandbytes, make sure old CUDA 10.0 versions are uninstalled and removed from $LD_LIBRARY_PATH variables.')
        return
    has_cublaslt = is_cublasLt_compatible(self.cc)
    if not has_cublaslt:
        make_cmd += '_nomatmul'
    self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
    self.add_log_entry('git clone https://github.com/TimDettmers/bitsandbytes.git')
    self.add_log_entry('cd bitsandbytes')
    self.add_log_entry(make_cmd)
    self.add_log_entry('python setup.py install')