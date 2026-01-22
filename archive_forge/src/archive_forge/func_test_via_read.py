from glance.tests import utils as test_utils
def test_via_read(self):
    fd = test_utils.FakeData(1024)
    data = []
    for i in range(0, 1025, 256):
        chunk = fd.read(256)
        data.append(chunk)
        if not chunk:
            break
    self.assertEqual(5, len(data))
    self.assertEqual(b'', data[-1])
    self.assertEqual(1024, len(b''.join(data)))