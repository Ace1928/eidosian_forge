from pprint import pformat
from six import iteritems
import re
@block_owner_deletion.setter
def block_owner_deletion(self, block_owner_deletion):
    """
        Sets the block_owner_deletion of this V1OwnerReference.
        If true, AND if the owner has the "foregroundDeletion" finalizer, then
        the owner cannot be deleted from the key-value store until this
        reference is removed. Defaults to false. To set this field, a user needs
        "delete" permission of the owner, otherwise 422 (Unprocessable Entity)
        will be returned.

        :param block_owner_deletion: The block_owner_deletion of this
        V1OwnerReference.
        :type: bool
        """
    self._block_owner_deletion = block_owner_deletion