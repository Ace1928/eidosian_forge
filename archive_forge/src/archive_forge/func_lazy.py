import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def lazy(func):

    @functools.wraps(func)
    def wrapper(self, *args):
        self._fill_lazy_properties()
        return func(self, *args)
    return wrapper