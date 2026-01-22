from glance_store import capabilities as caps
from glance_store.tests import base
def test_set_unset_capabilities(self):
    store = FakeStoreWithStaticCapabilities()
    self.assertFalse(store.is_capable(caps.BitMasks.WRITE_ACCESS))
    store.set_capabilities(caps.BitMasks.WRITE_ACCESS)
    self.assertTrue(store.is_capable(caps.BitMasks.WRITE_ACCESS))
    store.unset_capabilities(caps.BitMasks.WRITE_ACCESS)
    self.assertFalse(store.is_capable(caps.BitMasks.WRITE_ACCESS))
    cap_list = [caps.BitMasks.WRITE_ACCESS, caps.BitMasks.WRITE_OFFSET]
    store.set_capabilities(*cap_list)
    self.assertTrue(store.is_capable(*cap_list))
    store.unset_capabilities(*cap_list)
    self.assertFalse(store.is_capable(*cap_list))