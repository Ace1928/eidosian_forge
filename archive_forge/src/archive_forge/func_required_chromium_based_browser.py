import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
def required_chromium_based_browser(func):
    """A decorator to ensure that the client used is a chromium based
    browser."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.caps['browserName'].lower() not in ['firefox', 'safari'], 'This only currently works in Chromium based browsers'
        return func(self, *args, **kwargs)
    return wrapper