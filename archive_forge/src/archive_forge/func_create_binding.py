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
def create_binding(self, interface: type, to: Any=None, scope: Union['ScopeDecorator', Type['Scope'], None]=None) -> Binding:
    provider = self.provider_for(interface, to)
    scope = scope or getattr(to or interface, '__scope__', NoScope)
    if isinstance(scope, ScopeDecorator):
        scope = scope.scope
    return Binding(interface, provider, scope)