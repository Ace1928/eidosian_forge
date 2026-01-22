from glance_store import capabilities as caps
from glance_store.tests import base
def test_combined_capability_bits(self):
    check = caps.StoreCapability.contains
    check(caps.BitMasks.READ_OFFSET, caps.BitMasks.READ_ACCESS)
    check(caps.BitMasks.READ_CHUNK, caps.BitMasks.READ_ACCESS)
    check(caps.BitMasks.READ_RANDOM, caps.BitMasks.READ_CHUNK)
    check(caps.BitMasks.READ_RANDOM, caps.BitMasks.READ_OFFSET)
    check(caps.BitMasks.WRITE_OFFSET, caps.BitMasks.WRITE_ACCESS)
    check(caps.BitMasks.WRITE_CHUNK, caps.BitMasks.WRITE_ACCESS)
    check(caps.BitMasks.WRITE_RANDOM, caps.BitMasks.WRITE_CHUNK)
    check(caps.BitMasks.WRITE_RANDOM, caps.BitMasks.WRITE_OFFSET)
    check(caps.BitMasks.RW_ACCESS, caps.BitMasks.READ_ACCESS)
    check(caps.BitMasks.RW_ACCESS, caps.BitMasks.WRITE_ACCESS)
    check(caps.BitMasks.RW_OFFSET, caps.BitMasks.READ_OFFSET)
    check(caps.BitMasks.RW_OFFSET, caps.BitMasks.WRITE_OFFSET)
    check(caps.BitMasks.RW_CHUNK, caps.BitMasks.READ_CHUNK)
    check(caps.BitMasks.RW_CHUNK, caps.BitMasks.WRITE_CHUNK)
    check(caps.BitMasks.RW_RANDOM, caps.BitMasks.READ_RANDOM)
    check(caps.BitMasks.RW_RANDOM, caps.BitMasks.WRITE_RANDOM)