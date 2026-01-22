import multiprocessing.pool
import multiprocessing.util as util
from .queue import SimpleQueue
def clean_worker(*args, **kwargs):
    import gc
    multiprocessing.pool.worker(*args, **kwargs)
    gc.collect()