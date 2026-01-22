import re
from inspect import signature
from typing import Optional
import pytest
from sklearn.experimental import (
from sklearn.utils.discovery import all_displays, all_estimators, all_functions
def repr_errors(res, Klass=None, method: Optional[str]=None) -> str:
    """Pretty print original docstring and the obtained errors

    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    Klass : {Estimator, Display, None}
        estimator object or None
    method : str
        if estimator is not None, either the method name or None.

    Returns
    -------
    str
       String representation of the error.
    """
    if method is None:
        if hasattr(Klass, '__init__'):
            method = '__init__'
        elif Klass is None:
            raise ValueError('At least one of Klass, method should be provided')
        else:
            raise NotImplementedError
    if Klass is not None:
        obj = getattr(Klass, method)
        try:
            obj_signature = str(signature(obj))
        except TypeError:
            obj_signature = '\nParsing of the method signature failed, possibly because this is a property.'
        obj_name = Klass.__name__ + '.' + method
    else:
        obj_signature = ''
        obj_name = method
    msg = '\n\n' + '\n\n'.join([str(res['file']), obj_name + obj_signature, res['docstring'], '# Errors', '\n'.join((' - {}: {}'.format(code, message) for code, message in res['errors']))])
    return msg