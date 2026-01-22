from breezy import tests
from breezy.bzr import groupcompress
def inv_to_chk_inv(test, inv):
    """CHKInventory needs a backing VF, so we create one."""
    factory = groupcompress.make_pack_factory(True, True, 1)
    trans = test.get_transport('chk-inv')
    trans.ensure_base()
    vf = factory(trans)
    chk_inv = CHKInventory.from_inventory(vf, inv, maximum_size=100, search_key_name=b'hash-255-way')
    return chk_inv