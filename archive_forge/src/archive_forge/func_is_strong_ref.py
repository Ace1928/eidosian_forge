import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def is_strong_ref(self):
    return self.is_ref() and self.ref_type == 'strong'