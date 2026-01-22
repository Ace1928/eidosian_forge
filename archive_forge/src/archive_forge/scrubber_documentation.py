from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING

        If a dictionary is passed to this method, the method scrubs the dictionary of any
        sensitive data. The method calls itself recursively on any nested dictionaries (
        including dictionaries nested in lists) if self.recursive is True.
        This method does nothing if the parameter passed to it is not a dictionary.
        