from pprint import pformat
from six import iteritems
import re
@holder_identity.setter
def holder_identity(self, holder_identity):
    """
        Sets the holder_identity of this V1LeaseSpec.
        holderIdentity contains the identity of the holder of a current lease.

        :param holder_identity: The holder_identity of this V1LeaseSpec.
        :type: str
        """
    self._holder_identity = holder_identity