from __future__ import annotations
import logging # isort:skip
from functools import wraps
from inspect import Parameter, Signature
from ..util.deprecation import deprecated
from ._docstring import generate_docstring
from ._renderer import create_renderer
def glyph_method(glyphclass):

    def decorator(func):
        parameters = glyphclass.parameters()
        sigparams = [Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)] + [x[0] for x in parameters] + [Parameter('kwargs', Parameter.VAR_KEYWORD)]

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            if len(args) > len(glyphclass._args):
                raise TypeError(f'{func.__name__} takes {len(glyphclass._args)} positional argument but {len(args)} were given')
            for arg, param in zip(args, sigparams[1:]):
                kwargs[param.name] = arg
            if self.coordinates is not None:
                kwargs.setdefault('coordinates', self.coordinates)
            return create_renderer(glyphclass, self.plot, **kwargs)
        wrapped.__signature__ = Signature(parameters=sigparams)
        wrapped.__name__ = func.__name__
        wrapped.__doc__ = generate_docstring(glyphclass, parameters, func.__doc__)
        return wrapped
    return decorator