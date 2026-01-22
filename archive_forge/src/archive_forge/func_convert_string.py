import os
@classmethod
def convert_string(cls, param, value):
    if not isinstance(value, basestring):
        raise ValueError
    return value