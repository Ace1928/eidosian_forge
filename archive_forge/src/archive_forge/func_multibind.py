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
def multibind(self, interface: type, to: Any, scope: Union['ScopeDecorator', Type['Scope'], None]=None) -> None:
    """Creates or extends a multi-binding.

        A multi-binding contributes values to a list or to a dictionary. For example::

            binder.multibind(List[str], to=['some', 'strings'])
            binder.multibind(List[str], to=['other', 'strings'])
            injector.get(List[str])  # ['some', 'strings', 'other', 'strings']

            binder.multibind(Dict[str, int], to={'key': 11})
            binder.multibind(Dict[str, int], to={'other_key': 33})
            injector.get(Dict[str, int])  # {'key': 11, 'other_key': 33}

        .. versionchanged:: 0.17.0
            Added support for using `typing.Dict` and `typing.List` instances as interfaces.
            Deprecated support for `MappingKey`, `SequenceKey` and single-item lists and
            dictionaries as interfaces.

        :param interface: typing.Dict or typing.List instance to bind to.
        :param to: Instance, class to bind to, or an explicit :class:`Provider`
                subclass. Must provide a list or a dictionary, depending on the interface.
        :param scope: Optional Scope in which to bind.
        """
    if interface not in self._bindings:
        provider: ListOfProviders
        if isinstance(interface, dict) or (isinstance(interface, type) and issubclass(interface, dict)) or _get_origin(_punch_through_alias(interface)) is dict:
            provider = MapBindProvider()
        else:
            provider = MultiBindProvider()
        binding = self.create_binding(interface, provider, scope)
        self._bindings[interface] = binding
    else:
        binding = self._bindings[interface]
        provider = binding.provider
        assert isinstance(provider, ListOfProviders)
    provider.append(self.provider_for(interface, to))