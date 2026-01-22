import inspect
import os.path
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked
from inspect import getfullargspec
def generate_imports_tip_for_module(obj_to_complete, dir_comps=None, getattr=getattr, filter=lambda name: True):
    """
        @param obj_to_complete: the object from where we should get the completions
        @param dir_comps: if passed, we should not 'dir' the object and should just iterate those passed as kwonly_arg parameter
        @param getattr: the way to get kwonly_arg given object from the obj_to_complete (used for the completer)
        @param filter: kwonly_arg callable that receives the name and decides if it should be appended or not to the results
        @return: list of tuples, so that each tuple represents kwonly_arg completion with:
            name, doc, args, type (from the TYPE_* constants)
    """
    ret = []
    if dir_comps is None:
        dir_comps = dir_checked(obj_to_complete)
        if hasattr_checked(obj_to_complete, '__dict__'):
            dir_comps.append('__dict__')
        if hasattr_checked(obj_to_complete, '__class__'):
            dir_comps.append('__class__')
    get_complete_info = True
    if len(dir_comps) > 1000:
        get_complete_info = False
    dontGetDocsOn = (float, int, str, tuple, list, dict)
    dontGetattrOn = (dict, list, set, tuple)
    for d in dir_comps:
        if d is None:
            continue
        if not filter(d):
            continue
        args = ''
        try:
            try:
                if isinstance(obj_to_complete, dontGetattrOn):
                    raise Exception('Since python 3.9, e.g. "dict[str]" will return a dict that\'s only supposed to take strings. Interestingly, e.g. dict["val"] is also valid and presumably represents a dict that only takes keys that are "val". This breaks our check for class attributes.')
                obj = getattr(obj_to_complete.__class__, d)
            except:
                obj = getattr(obj_to_complete, d)
        except:
            ret.append((d, '', args, TYPE_BUILTIN))
        else:
            if get_complete_info:
                try:
                    retType = TYPE_BUILTIN
                    getDoc = True
                    for class_ in dontGetDocsOn:
                        if isinstance(obj, class_):
                            getDoc = False
                            break
                    doc = ''
                    if getDoc:
                        try:
                            doc = inspect.getdoc(obj)
                            if doc is None:
                                doc = ''
                        except:
                            doc = ''
                    if inspect.ismethod(obj) or inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.isroutine(obj):
                        try:
                            args, vargs, kwargs, defaults, kwonly_args, kwonly_defaults = getargspec(obj)
                            args = args[:]
                            for kwonly_arg in kwonly_args:
                                default = kwonly_defaults.get(kwonly_arg, _SENTINEL)
                                if default is not _SENTINEL:
                                    args.append('%s=%s' % (kwonly_arg, default))
                                else:
                                    args.append(str(kwonly_arg))
                            args = '(%s)' % ', '.join(args)
                        except TypeError:
                            args, doc = signature_from_docstring(doc, getattr(obj, '__name__', None))
                        retType = TYPE_FUNCTION
                    elif inspect.isclass(obj):
                        retType = TYPE_CLASS
                    elif inspect.ismodule(obj):
                        retType = TYPE_IMPORT
                    else:
                        retType = TYPE_ATTR
                    ret.append((d, doc, args, retType))
                except:
                    ret.append((d, '', args, TYPE_BUILTIN))
            else:
                if inspect.ismethod(obj) or inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.isroutine(obj):
                    retType = TYPE_FUNCTION
                elif inspect.isclass(obj):
                    retType = TYPE_CLASS
                elif inspect.ismodule(obj):
                    retType = TYPE_IMPORT
                else:
                    retType = TYPE_ATTR
                ret.append((d, '', str(args), retType))
    return ret