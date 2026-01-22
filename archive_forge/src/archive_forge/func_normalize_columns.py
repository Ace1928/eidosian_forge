import collections
import re
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
def normalize_columns(row, expandtabs=False):
    results = []
    for value in row:
        text = coerce_string(value)
        if expandtabs:
            text = text.expandtabs()
        results.append(text)
    return results