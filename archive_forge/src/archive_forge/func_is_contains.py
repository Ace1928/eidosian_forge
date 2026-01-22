import abc
import copy
from neutron_lib import exceptions
@property
def is_contains(self):
    return bool(getattr(self, 'contains', False))