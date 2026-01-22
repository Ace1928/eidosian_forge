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
def set_memory_allocator(self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False):
    """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.

        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory reuse and reduce page fault to improve performance.
        """
    if enable_tcmalloc and enable_jemalloc:
        raise RuntimeError('Unable to enable TCMalloc and JEMalloc at the same time.')
    if enable_tcmalloc:
        find_tc = self.add_lib_preload(lib_type='tcmalloc')
        if not find_tc:
            msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge gperftools" to install {{0}}'
            logger.warning(msg.format('TCmalloc', 'tcmalloc'))
        else:
            logger.info('Use TCMalloc memory allocator')
    elif enable_jemalloc:
        find_je = self.add_lib_preload(lib_type='jemalloc')
        if not find_je:
            msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge jemalloc" to install {{0}}'
            logger.warning(msg.format('Jemalloc', 'jemalloc'))
        else:
            logger.info('Use JeMalloc memory allocator')
            self.set_env('MALLOC_CONF', 'oversize_threshold:1,background_thread:true,metadata_thp:auto')
    elif use_default_allocator:
        pass
    else:
        find_tc = self.add_lib_preload(lib_type='tcmalloc')
        if find_tc:
            logger.info('Use TCMalloc memory allocator')
            return
        find_je = self.add_lib_preload(lib_type='jemalloc')
        if find_je:
            logger.info('Use JeMalloc memory allocator')
            return
        logger.warning('Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib\n                            or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or\n                           %s/.local/lib/ so the LD_PRELOAD environment variable will not be set.\n                           This may drop the performance', expanduser('~'))