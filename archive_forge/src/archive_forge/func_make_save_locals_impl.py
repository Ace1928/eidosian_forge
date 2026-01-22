import sys
def make_save_locals_impl():
    """
    Factory for the 'save_locals_impl' method. This may seem like a complicated pattern but it is essential that the method is created at
    module load time. Inner imports after module load time would cause an occasional debugger deadlock due to the importer lock and debugger
    lock being taken in different order in  different threads.
    """
    try:
        if '__pypy__' in sys.builtin_module_names:
            import __pypy__
            save_locals = __pypy__.locals_to_fast
    except:
        pass
    else:
        if '__pypy__' in sys.builtin_module_names:

            def save_locals_pypy_impl(frame):
                save_locals(frame)
            return save_locals_pypy_impl
    try:
        import ctypes
        locals_to_fast = ctypes.pythonapi.PyFrame_LocalsToFast
    except:
        pass
    else:

        def save_locals_ctypes_impl(frame):
            locals_to_fast(ctypes.py_object(frame), ctypes.c_int(0))
        return save_locals_ctypes_impl
    return None