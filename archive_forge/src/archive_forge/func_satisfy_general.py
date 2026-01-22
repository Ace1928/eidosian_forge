import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def satisfy_general(self, func):
    if not hasattr(func, '__call__'):
        raise TypeError('General caveat verifiers must be callable.')
    self.callbacks.append(func)