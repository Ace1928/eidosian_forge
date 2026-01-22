from pprint import pformat
from six import iteritems
import re
@secret_file.setter
def secret_file(self, secret_file):
    """
        Sets the secret_file of this V1CephFSPersistentVolumeSource.
        Optional: SecretFile is the path to key ring for User, default is
        /etc/ceph/user.secret More info:
        https://releases.k8s.io/HEAD/examples/volumes/cephfs/README.md#how-to-use-it

        :param secret_file: The secret_file of this
        V1CephFSPersistentVolumeSource.
        :type: str
        """
    self._secret_file = secret_file