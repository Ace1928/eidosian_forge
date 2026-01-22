import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def is_optional_pointer(self):
    return self.is_optional() and (not self.value) and (self.key.type == StringType or self.key.ref_table_name)