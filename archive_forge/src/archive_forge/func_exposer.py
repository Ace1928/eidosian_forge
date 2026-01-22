import itertools
from typing import (
from zope.interface import implementer
from twisted.web.error import (
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
def exposer(thunk: Callable[..., object]) -> Expose:
    expose = Expose()
    expose.__doc__ = thunk.__doc__
    return expose