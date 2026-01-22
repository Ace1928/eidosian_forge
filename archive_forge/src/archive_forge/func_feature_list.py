import ctypes
import collections
from .base import _LIB, check_call
def feature_list():
    """Check the library for compile-time features. The list of features are maintained in libinfo.h and libinfo.cc

    Returns
    -------
    list
        List of :class:`.Feature` objects
    """
    lib_features_c_array = ctypes.POINTER(Feature)()
    lib_features_size = ctypes.c_size_t()
    check_call(_LIB.MXLibInfoFeatures(ctypes.byref(lib_features_c_array), ctypes.byref(lib_features_size)))
    features = [lib_features_c_array[i] for i in range(lib_features_size.value)]
    return features