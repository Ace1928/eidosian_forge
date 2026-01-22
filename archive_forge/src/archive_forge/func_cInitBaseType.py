import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def cInitBaseType(self, prefix, prereqs):
    init = ['.type = %s,' % self.toAtomicType()]
    if self.enum:
        datum_name = '%s_enum' % prefix
        init += ['.enum_ = &%s,' % datum_name]
        prereqs += self.enum.cDeclareDatum(datum_name)
    if self.type == IntegerType:
        if self.min is None:
            low = 'INT64_MIN'
        else:
            low = 'INT64_C(%d)' % self.min
        if self.max is None:
            high = 'INT64_MAX'
        else:
            high = 'INT64_C(%d)' % self.max
        init.append('.integer = { .min = %s, .max = %s },' % (low, high))
    elif self.type == RealType:
        if self.min is None:
            low = '-DBL_MAX'
        else:
            low = self.min
        if self.max is None:
            high = 'DBL_MAX'
        else:
            high = self.max
        init.append('.real = { .min = %s, .max = %s },' % (low, high))
    elif self.type == StringType:
        if self.min is None:
            low = 0
        else:
            low = self.min_length
        if self.max is None:
            high = 'UINT_MAX'
        else:
            high = self.max_length
        init.append('.string = { .minLen = %s, .maxLen = %s },' % (low, high))
    elif self.type == UuidType:
        if self.ref_table_name is not None:
            init.append('.uuid = { .refTableName = "%s", .refType = OVSDB_REF_%s },' % (escapeCString(self.ref_table_name), self.ref_type.upper()))
    return init