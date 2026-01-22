from __future__ import print_function, absolute_import
from textwrap import dedent
from shibokensupport.signature import inspect, typing
from shibokensupport.signature.mapping import ellipsis
from shibokensupport.signature.lib.tool import SimpleNamespace
def define_nameless_parameter():
    """
    Create Nameless Parameters

    A nameless parameter has a reduced string representation.
    This is done by cloning the parameter type and overwriting its
    __str__ method. The inner structure is still a valid parameter.
    """

    def __str__(self):
        klass = self.__class__
        self.__class__ = P
        txt = P.__str__(self)
        self.__class__ = klass
        txt = txt[txt.index(':') + 1:].strip() if ':' in txt else txt
        return txt
    P = inspect.Parameter
    newname = 'NamelessParameter'
    bases = P.__bases__
    body = dict(P.__dict__)
    if '__slots__' in body:
        for name in body['__slots__']:
            del body[name]
    body['__str__'] = __str__
    return type(newname, bases, body)