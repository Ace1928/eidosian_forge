import os
import sys
import re
def namerepl(mobj):
    name = mobj.group(1)
    return rules.get(name, (k + 1) * [name])[k]