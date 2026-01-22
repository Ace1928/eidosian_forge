from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
def find_closest_generator(i):
    """ Return a generator close to line 'i', or None if none was fonud. """
    for generator, lineno in line_generators.items():
        if lineno < i and i - lineno < self.REUSE_GENERATOR_MAX_DISTANCE:
            return generator