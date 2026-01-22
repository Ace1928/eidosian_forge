import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def release_data(key, state, delete=True):
    """Remove data from temporary storage
    See Also
    --------
    finish_task
    """
    if key in state['waiting_data']:
        assert not state['waiting_data'][key]
        del state['waiting_data'][key]
    state['released'].add(key)
    if delete:
        del state['cache'][key]