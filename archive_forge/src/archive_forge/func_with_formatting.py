import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@wraps(func)
def with_formatting(*args, **kwargs):
    print()
    print(SPACER)
    print('Using %s' % func.__name__)
    print(SPACER)
    return func(*args, **kwargs)