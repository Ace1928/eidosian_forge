from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def define_interfaces(type, interfaces):
    if callable(interfaces):
        interfaces = interfaces()
    if interfaces is None:
        interfaces = []
    assert isinstance(interfaces, (list, tuple)), '{} interfaces must be a list/tuple or a function which returns a list/tuple.'.format(type)
    for interface in interfaces:
        assert isinstance(interface, GraphQLInterfaceType), '{} may only implement Interface types, it cannot implement: {}.'.format(type, interface)
        if not callable(interface.resolve_type):
            assert callable(type.is_type_of), 'Interface Type {} does not provide a "resolve_type" function and implementing Type {} does not provide a "is_type_of" function. There is no way to resolve this implementing type during execution.'.format(interface, type)
    return interfaces