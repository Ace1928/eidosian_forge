import sys
import inspect
def new_wrapper(wrapper, model):
    """
    An improvement over functools.update_wrapper. The wrapper is a generic
    callable object. It works by generating a copy of the wrapper with the
    right signature and by updating the copy, not the original.
    Moreovoer, 'model' can be a dictionary with keys 'name', 'doc', 'module',
    'dict', 'defaults'.
    """
    if isinstance(model, dict):
        infodict = model
    else:
        infodict = getinfo(model)
    assert not '_wrapper_' in infodict['argnames'], '"_wrapper_" is a reserved argument name!'
    src = 'lambda %(signature)s: _wrapper_(%(signature)s)' % infodict
    funcopy = eval(src, dict(_wrapper_=wrapper))
    return update_wrapper(funcopy, model, infodict)