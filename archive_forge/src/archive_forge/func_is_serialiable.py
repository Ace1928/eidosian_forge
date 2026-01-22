import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def is_serialiable(obj):
    """Check if *obj* can be serialized.

    Parameters
    ----------
    obj : object

    Returns
    --------
    can_serialize : bool
    """
    with io.BytesIO() as fout:
        pickler = NumbaPickler(fout)
        try:
            pickler.dump(obj)
        except pickle.PicklingError:
            return False
        else:
            return True