import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def initCDefault(self, var, is_optional):
    if self.ref_table_name:
        return '%s = NULL;' % var
    elif self.type == StringType and (not is_optional):
        return '%s = "";' % var
    else:
        pattern = {IntegerType: '%s = 0;', RealType: '%s = 0.0;', UuidType: 'uuid_zero(&%s);', BooleanType: '%s = false;', StringType: '%s = NULL;'}[self.type]
        return pattern % var