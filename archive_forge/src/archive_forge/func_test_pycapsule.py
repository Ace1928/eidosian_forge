import dill
import warnings
def test_pycapsule():
    name = ctypes.create_string_buffer(b'dill._testcapsule')
    capsule = dill._dill._PyCapsule_New(ctypes.cast(dill._dill._PyCapsule_New, ctypes.c_void_p), name, None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dill.copy(capsule)
    dill._testcapsule = capsule
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dill.copy(capsule)
    dill._testcapsule = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', dill.PicklingWarning)
            dill.copy(capsule)
    except dill.UnpicklingError:
        pass
    else:
        raise AssertionError('Expected a different error')