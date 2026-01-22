import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
def reregister(self, class_):
    """Register a type which may already have been registered.
        """
    self._unregister(class_)
    return self.register(class_)