from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@classmethod
def from_other(cls, ori, **kwargs):
    """ Creates a new instance with an existing one as a template.

        Parameters
        ----------
        ori : SymbolicSys instance
        \\*\\*kwargs:
            Keyword arguments used to create the new instance.

        Returns
        -------
        A new instance of the class.

        """
    for k in cls._attrs_to_copy + ('params', 'roots', 'init_indep', 'init_dep'):
        if k not in kwargs:
            val = getattr(ori, k)
            if val is not None:
                kwargs[k] = val
    if 'lower_bounds' not in kwargs and getattr(ori, 'lower_bounds') is not None:
        kwargs['lower_bounds'] = ori.lower_bounds
    if 'upper_bounds' not in kwargs and getattr(ori, 'upper_bounds') is not None:
        kwargs['upper_bounds'] = ori.upper_bounds
    if len(ori.pre_processors) > 0:
        if 'pre_processors' not in kwargs:
            kwargs['pre_processors'] = []
        kwargs['pre_processors'] = kwargs['pre_processors'] + ori.pre_processors
    if len(ori.post_processors) > 0:
        if 'post_processors' not in kwargs:
            kwargs['post_processors'] = []
        kwargs['post_processors'] = ori.post_processors + kwargs['post_processors']
    if 'dep_exprs' not in kwargs:
        kwargs['dep_exprs'] = zip(ori.dep, ori.exprs)
    if 'indep' not in kwargs:
        kwargs['indep'] = ori.indep
    instance = cls(**kwargs)
    for attr in ori._attrs_to_copy:
        if attr not in cls._attrs_to_copy:
            setattr(instance, attr, getattr(ori, attr))
    return instance