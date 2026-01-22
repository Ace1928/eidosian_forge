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
def set_multi_thread_and_allocator(self, ncores_per_instance, disable_iomp=False, set_kmp_affinity=True, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False):
    """
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.

        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        """
    self.set_memory_allocator(enable_tcmalloc, enable_jemalloc, use_default_allocator)
    self.set_env('OMP_NUM_THREADS', str(ncores_per_instance))
    if not disable_iomp:
        find_iomp = self.add_lib_preload(lib_type='iomp5')
        if not find_iomp:
            msg = f'{self.msg_lib_notfound} you can use "conda install mkl" to install {{0}}'
            logger.warning(msg.format('iomp', 'iomp5'))
        else:
            logger.info('Using Intel OpenMP')
            if set_kmp_affinity:
                self.set_env('KMP_AFFINITY', 'granularity=fine,compact,1,0')
            self.set_env('KMP_BLOCKTIME', '1')
    self.log_env_var('LD_PRELOAD')