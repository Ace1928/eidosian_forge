import os
from threading import Lock
import warnings
from .mmap_dict import mmap_key, MmapedDict
def get_value_class():
    if 'prometheus_multiproc_dir' in os.environ or 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
        return MultiProcessValue()
    else:
        return MutexValue