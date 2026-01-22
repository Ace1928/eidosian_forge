import re
import distutils.version
import operator
def satisfied_by(self, version):
    """True if version is compatible with all the predicates in self.
        The parameter version must be acceptable to the StrictVersion
        constructor.  It may be either a string or StrictVersion.
        """
    for cond, ver in self.pred:
        if not compmap[cond](version, ver):
            return False
    return True