import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
def ssf(self):
    """Simple symbol form."""
    ssfs = [n.ssf() if isinstance(n, QN) else n for n in self.qn]
    ssf_string = ''
    for i in range(0, len(self.qn) - 1):
        if self.has_subscript():
            delimiter = '_sub_'
        else:
            delimiter = '_'
        ssf_string += ssfs[i] + delimiter
    return ssf_string + ssfs[-1]