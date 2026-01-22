from breezy import errors
from breezy.bzr import knit
from breezy.tests.per_repository_reference import \
def test_ordered_no_closure(self):
    self.make_complex_split()
    keys = [(b'f-id', bytes([r])) for r in bytearray(b'ABCDEG')]
    alt_1 = [(b'f-id', bytes([r])) for r in bytearray(b'ACBDEG')]
    alt_2 = [(b'f-id', bytes([r])) for r in bytearray(b'ABCEDG')]
    alt_3 = [(b'f-id', bytes([r])) for r in bytearray(b'ACBEDG')]
    alt_4 = [(b'f-id', bytes([r])) for r in bytearray(b'ACEBDG')]
    self.stacked_repo.lock_read()
    self.addCleanup(self.stacked_repo.unlock)
    stream = self.stacked_repo.texts.get_record_stream(keys, 'topological', False)
    record_keys = []
    for record in stream:
        if record.storage_kind == 'absent':
            raise ValueError('absent record: {}'.format(record.key))
        record_keys.append(record.key)
    self.assertIn(record_keys, (keys, alt_1, alt_2, alt_3, alt_4))