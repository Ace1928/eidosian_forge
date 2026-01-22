import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def cInitType(self, prefix, prereqs):
    init = ['.key = {']
    init += ['   ' + x for x in self.key.cInitBaseType(prefix + '_key', prereqs)]
    init += ['},']
    if self.value:
        init += ['.value = {']
        init += ['    ' + x for x in self.value.cInitBaseType(prefix + '_value', prereqs)]
        init += ['},']
    else:
        init.append('.value = OVSDB_BASE_VOID_INIT,')
    init.append('.n_min = %s,' % self.n_min)
    if self.n_max == sys.maxsize:
        n_max = 'UINT_MAX'
    else:
        n_max = self.n_max
    init.append('.n_max = %s,' % n_max)
    return init