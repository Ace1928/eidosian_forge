import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def provider_for(self, interface: Any, to: Any=None) -> Provider:
    base_type = _punch_through_alias(interface)
    origin = _get_origin(base_type)
    if interface is Any:
        raise TypeError('Injecting Any is not supported')
    elif _is_specialization(interface, ProviderOf):
        target, = interface.__args__
        if to is not None:
            raise Exception('ProviderOf cannot be bound to anything')
        return InstanceProvider(ProviderOf(self.injector, target))
    elif isinstance(to, Provider):
        return to
    elif isinstance(to, (types.FunctionType, types.LambdaType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
        return CallableProvider(to)
    elif issubclass(type(to), type):
        return ClassProvider(cast(type, to))
    elif isinstance(interface, BoundKey):

        def proxy(injector: Injector) -> Any:
            binder = injector.binder
            kwarg_providers = {name: binder.provider_for(None, provider) for name, provider in interface.kwargs.items()}
            kwargs = {name: provider.get(injector) for name, provider in kwarg_providers.items()}
            return interface.interface(**kwargs)
        return CallableProvider(inject(proxy))
    elif _is_specialization(interface, AssistedBuilder):
        target, = interface.__args__
        builder = interface(self.injector, target)
        return InstanceProvider(builder)
    elif origin is None and isinstance(base_type, (tuple, type)) and (interface is not Any) and isinstance(to, base_type) or (origin in {dict, list} and isinstance(to, origin)):
        return InstanceProvider(to)
    elif issubclass(type(base_type), type) or isinstance(base_type, (tuple, list)):
        if to is not None:
            return InstanceProvider(to)
        return ClassProvider(base_type)
    else:
        raise UnknownProvider("couldn't determine provider for %r to %r" % (interface, to))