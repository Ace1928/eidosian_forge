from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from openstack import exceptions
from openstack.image.iterable_chunked_file import IterableChunkedFile
def generate_signature(self, file_obj):
    if not self.private_key:
        raise exceptions.SDKException('private_key not set')
    file_obj.seek(0)
    chunked_file = IterableChunkedFile(file_obj)
    for chunk in chunked_file:
        self.hasher.update(chunk)
    file_obj.seek(0)
    digest = self.hasher.finalize()
    signature = self.private_key.sign(digest, self.padding, utils.Prehashed(self.hash))
    return signature