import re
def floatstr(o, allow_nan=self.allow_nan, _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
    if o != o:
        text = 'NaN'
    elif o == _inf:
        text = 'Infinity'
    elif o == _neginf:
        text = '-Infinity'
    else:
        return _repr(o)
    if not allow_nan:
        raise ValueError('Out of range float values are not JSON compliant: ' + repr(o))
    return text