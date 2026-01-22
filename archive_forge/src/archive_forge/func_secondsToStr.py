from __future__ import (absolute_import, division, print_function)
import collections
import time
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.six.moves import reduce
def secondsToStr(t):

    def rediv(ll, b):
        return list(divmod(ll[0], b)) + ll[1:]
    return '%d:%02d:%02d.%03d' % tuple(reduce(rediv, [[t * 1000], 1000, 60, 60]))