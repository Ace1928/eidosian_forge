import re
import string
import types
def js_output(self, attrs=None):
    """Return a string suitable for JavaScript."""
    result = []
    items = sorted(self.items())
    for key, value in items:
        result.append(value.js_output(attrs))
    return _nulljoin(result)