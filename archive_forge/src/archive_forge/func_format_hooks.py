import builtins
import types
import sys
from inspect import getmembers
from webob.exc import HTTPFound
from .util import iscontroller, _cfg
def format_hooks(self, hooks):
    """
        Tries to format the hook objects to be more readable
        Specific to Pecan (not available in the request object)
        """
    str_hooks = [str(i).split()[0].strip('<') for i in hooks]
    return [i.split('.')[-1] for i in str_hooks if '.' in i]