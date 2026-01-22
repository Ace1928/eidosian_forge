import sys
def update_module_path():
    import os
    global _orig_sys_path
    _orig_sys_path = sys.path[:]
    for path in __path__:
        if path in sys.path:
            sys.path.remove(path)
        path = os.path.abspath(path)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)