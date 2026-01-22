from .error import *
from .nodes import *
import collections.abc, datetime, base64, binascii, re, sys, types
def check_state_key(self, key):
    """Block special attributes/methods from being set in a newly created
        object, to prevent user-controlled methods from being called during
        deserialization"""
    if self.get_state_keys_blacklist_regexp().match(key):
        raise ConstructorError(None, None, "blacklisted key '%s' in instance state found" % (key,), None)