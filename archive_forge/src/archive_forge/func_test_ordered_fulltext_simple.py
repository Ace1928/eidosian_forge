from breezy import errors
from breezy.bzr import knit
from breezy.tests.per_repository_reference import \
def test_ordered_fulltext_simple(self):
    self.make_simple_split()
    keys = [(b'f-id', bytes([r])) for r in bytearray(b'ABCDF')]
    alt_1 = [(b'f-id', bytes([r])) for r in bytearray(b'ACBDF')]
    self.stacked_repo.lock_read()
    self.addCleanup(self.stacked_repo.unlock)
    stream = self.stacked_repo.texts.get_record_stream(keys, 'topological', True)
    record_keys = []
    for record in stream:
        if record.storage_kind == 'absent':
            raise ValueError('absent record: {}'.format(record.key))
        record_keys.append(record.key)
    self.assertIn(record_keys, (keys, alt_1))