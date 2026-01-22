import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def cDeclComment(self):
    if self.n_min == 1 and self.n_max == 1 and (self.key.type == StringType):
        return '\t/* Always nonnull. */'
    else:
        return ''