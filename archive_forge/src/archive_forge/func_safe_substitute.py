import _string
import re as _re
from collections import ChainMap as _ChainMap
def safe_substitute(self, mapping=_sentinel_dict, /, **kws):
    if mapping is _sentinel_dict:
        mapping = kws
    elif kws:
        mapping = _ChainMap(kws, mapping)

    def convert(mo):
        named = mo.group('named') or mo.group('braced')
        if named is not None:
            try:
                return str(mapping[named])
            except KeyError:
                return mo.group()
        if mo.group('escaped') is not None:
            return self.delimiter
        if mo.group('invalid') is not None:
            return mo.group()
        raise ValueError('Unrecognized named group in pattern', self.pattern)
    return self.pattern.sub(convert, self.template)