from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_kern_dist(self, c1, c2):
    """
        Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.
        """
    name1, name2 = (self.get_name_char(c1), self.get_name_char(c2))
    return self.get_kern_dist_from_name(name1, name2)