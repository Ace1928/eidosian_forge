import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def get_async(apply_async, num_workers, dsk, result, cache=None, get_id=default_get_id, rerun_exceptions_locally=None, pack_exception=default_pack_exception, raise_exception=reraise, callbacks=None, dumps=identity, loads=identity, **kwargs):
    """Asynchronous get function
    This is a general version of various asynchronous schedulers for dask.  It
    takes a an apply_async function as found on Pool objects to form a more
    specific ``get`` method that walks through the dask array with parallel
    workers, avoiding repeat computation and minimizing memory use.
    Parameters
    ----------
    apply_async : function
        Asynchronous apply function as found on Pool or ThreadPool
    num_workers : int
        The number of active tasks we should have at any one time
    dsk : dict
        A dask dictionary specifying a workflow
    result : key or list of keys
        Keys corresponding to desired data
    cache : dict-like, optional
        Temporary storage of results
    get_id : callable, optional
        Function to return the worker id, takes no arguments. Examples are
        `threading.current_thread` and `multiprocessing.current_process`.
    rerun_exceptions_locally : bool, optional
        Whether to rerun failing tasks in local process to enable debugging
        (False by default)
    pack_exception : callable, optional
        Function to take an exception and ``dumps`` method, and return a
        serialized tuple of ``(exception, traceback)`` to send back to the
        scheduler. Default is to just raise the exception.
    raise_exception : callable, optional
        Function that takes an exception and a traceback, and raises an error.
    dumps: callable, optional
        Function to serialize task data and results to communicate between
        worker and parent.  Defaults to identity.
    loads: callable, optional
        Inverse function of `dumps`.  Defaults to identity.
    callbacks : tuple or list of tuples, optional
        Callbacks are passed in as tuples of length 5. Multiple sets of
        callbacks may be passed in as a list of tuples. For more information,
        see the dask.diagnostics documentation.
    See Also
    --------
    threaded.get
    """
    queue = Queue()
    if isinstance(result, list):
        result_flat = set(flatten(result))
    else:
        result_flat = {result}
    results = set(result_flat)
    dsk = dict(dsk)
    with local_callbacks(callbacks) as callbacks:
        _, _, pretask_cbs, posttask_cbs, _ = unpack_callbacks(callbacks)
        started_cbs = []
        succeeded = False
        state = {}
        try:
            for cb in callbacks:
                if cb[0]:
                    cb[0](dsk)
                started_cbs.append(cb)
            keyorder = order(dsk)
            state = start_state_from_dask(dsk, cache=cache, sortkey=keyorder.get)
            for _, start_state, _, _, _ in callbacks:
                if start_state:
                    start_state(dsk, state)
            if rerun_exceptions_locally is None:
                rerun_exceptions_locally = config.get('rerun_exceptions_locally', False)
            if state['waiting'] and (not state['ready']):
                raise ValueError('Found no accessible jobs in dask')

            def fire_task():
                """Fire off a task to the thread pool"""
                key = state['ready'].pop()
                state['running'].add(key)
                for f in pretask_cbs:
                    f(key, dsk, state)
                data = {dep: state['cache'][dep] for dep in get_dependencies(dsk, key)}
                apply_async(execute_task, args=(key, dumps((dsk[key], data)), dumps, loads, get_id, pack_exception), callback=queue.put)
            while state['ready'] and len(state['running']) < num_workers:
                fire_task()
            while state['waiting'] or state['ready'] or state['running']:
                key, res_info, failed = queue_get(queue)
                if failed:
                    exc, tb = loads(res_info)
                    if rerun_exceptions_locally:
                        data = {dep: state['cache'][dep] for dep in get_dependencies(dsk, key)}
                        task = dsk[key]
                        _execute_task(task, data)
                    else:
                        raise_exception(exc, tb)
                res, worker_id = loads(res_info)
                state['cache'][key] = res
                finish_task(dsk, key, state, results, keyorder.get)
                for f in posttask_cbs:
                    f(key, res, dsk, state, worker_id)
                while state['ready'] and len(state['running']) < num_workers:
                    fire_task()
            succeeded = True
        finally:
            for _, _, _, _, finish in started_cbs:
                if finish:
                    finish(dsk, state, not succeeded)
    return nested_get(result, state['cache'])