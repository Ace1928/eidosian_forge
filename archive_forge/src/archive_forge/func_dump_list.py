import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
def dump_list(self, v):
    t = []
    retval = '['
    for u in v:
        t.append(self.dump_value(u))
    while t != []:
        s = []
        for u in t:
            if isinstance(u, list):
                for r in u:
                    s.append(r)
            else:
                retval += ' ' + unicode(u) + self.separator
        t = s
    retval += ']'
    return retval