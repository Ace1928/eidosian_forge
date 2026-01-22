import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def toAtomicType(self):
    return 'OVSDB_TYPE_%s' % self.type.to_string().upper()