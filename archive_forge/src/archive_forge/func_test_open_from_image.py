import h5py
from h5py import h5f, h5p
from .common import ut, TestCase
def test_open_from_image(self):
    from binascii import a2b_base64
    from zlib import decompress
    compressed_image = 'eJzr9HBx4+WS4mIAAQ4OBhYGAQZk8B8KKjhQ+TD5BCjNCKU7oPQKJpg4I1hOAiouCDUfXV1IkKsrSPV/NACzx4AFQnMwjIKRCDxcHQNAdASUD0ulJ5hQ1ZWkFpeAaFh69KDQXkYGNohZjDA+JCUzMkIEmKHqELQAWKkAByytOoBJViAPJM7ExATWyAE0B8RgZkyAJmlYDoEAIahukJoNU6+HMTA0UOgT6oBgP38XUI6G5UMFZrzKR8EoGAUjGMDKYVgxDSsuAHcfMK8='
    image = decompress(a2b_base64(compressed_image))
    fid = h5f.open_file_image(image)
    f = h5py.File(fid)
    self.assertTrue('test' in f)