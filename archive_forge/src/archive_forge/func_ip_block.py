from pprint import pformat
from six import iteritems
import re
@ip_block.setter
def ip_block(self, ip_block):
    """
        Sets the ip_block of this V1beta1NetworkPolicyPeer.
        IPBlock defines policy on a particular IPBlock. If this field is set
        then neither of the other fields can be.

        :param ip_block: The ip_block of this V1beta1NetworkPolicyPeer.
        :type: V1beta1IPBlock
        """
    self._ip_block = ip_block