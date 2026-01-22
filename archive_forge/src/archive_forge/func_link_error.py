from abc import ABCMeta, abstractmethod
from collections.abc import Callable
@abstractmethod
def link_error(self, errback):
    pass