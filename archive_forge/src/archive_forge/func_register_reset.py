from functools import partial
def register_reset(func):
    """register a function to be called by rl_config._reset"""
    _registered_resets[:] = [x for x in _registered_resets if x()]
    L = [x for x in _registered_resets if x() is func]
    if L:
        return
    from weakref import ref
    _registered_resets.append(ref(func))