import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def globalvars(func, recurse=True, builtin=False):
    """get objects defined in global scope that are referred to by func

    return a dict of {name:object}"""
    if ismethod(func):
        func = func.__func__
    if isfunction(func):
        globs = vars(getmodule(sum)).copy() if builtin else {}
        orig_func, func = (func, set())
        for obj in orig_func.__closure__ or {}:
            try:
                cell_contents = obj.cell_contents
            except ValueError:
                pass
            else:
                _vars = globalvars(cell_contents, recurse, builtin) or {}
                func.update(_vars)
                globs.update(_vars)
        globs.update(orig_func.__globals__ or {})
        if not recurse:
            func.update(orig_func.__code__.co_names)
        else:
            func.update(nestedglobals(orig_func.__code__))
            for key in func.copy():
                nested_func = globs.get(key)
                if nested_func is orig_func:
                    continue
                func.update(globalvars(nested_func, True, builtin))
    elif iscode(func):
        globs = vars(getmodule(sum)).copy() if builtin else {}
        if not recurse:
            func = func.co_names
        else:
            orig_func = func.co_name
            func = set(nestedglobals(func))
            for key in func.copy():
                if key is orig_func:
                    continue
                nested_func = globs.get(key)
                func.update(globalvars(nested_func, True, builtin))
    else:
        return {}
    return dict(((name, globs[name]) for name in func if name in globs))