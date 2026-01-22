import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def get_doc_object(obj, what=None, doc=None, config=None):
    if what is None:
        if inspect.isclass(obj):
            what = 'class'
        elif inspect.ismodule(obj):
            what = 'module'
        elif isinstance(obj, Callable):
            what = 'function'
        else:
            what = 'object'
    if config is None:
        config = {}
    if what == 'class':
        return ClassDoc(obj, func_doc=FunctionDoc, doc=doc, config=config)
    elif what in ('function', 'method'):
        return FunctionDoc(obj, doc=doc, config=config)
    else:
        if doc is None:
            doc = pydoc.getdoc(obj)
        return ObjDoc(obj, doc, config=config)