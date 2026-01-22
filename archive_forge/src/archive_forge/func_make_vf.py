from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def make_vf(self):
    t = self.get_transport('')
    factory = groupcompress.make_pack_factory(True, True, 1)
    return factory(t)