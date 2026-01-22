import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
def wrap_non_picklable_objects(obj, keep_wrapper=True):
    """Wrapper for non-picklable object to use cloudpickle to serialize them.

    Note that this wrapper tends to slow down the serialization process as it
    is done with cloudpickle which is typically slower compared to pickle. The
    proper way to solve serialization issues is to avoid defining functions and
    objects in the main scripts and to implement __reduce__ functions for
    complex classes.
    """
    if inspect.isclass(obj):

        class CloudpickledClassWrapper(CloudpickledObjectWrapper):

            def __init__(self, *args, **kwargs):
                self._obj = obj(*args, **kwargs)
                self._keep_wrapper = keep_wrapper
        CloudpickledClassWrapper.__name__ = obj.__name__
        return CloudpickledClassWrapper
    return _wrap_non_picklable_objects(obj, keep_wrapper=keep_wrapper)