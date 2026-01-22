import sys
import fixtures
from functools import wraps
@transport_url.setter
def transport_url(self, value):
    self.conf.set_override('transport_url', value)