from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
@abstractmethod
def feed_key(self, key):
    pass