from pprint import pformat
from six import iteritems
import re
@described_object.setter
def described_object(self, described_object):
    """
        Sets the described_object of this V2beta2ObjectMetricStatus.

        :param described_object: The described_object of this
        V2beta2ObjectMetricStatus.
        :type: V2beta2CrossVersionObjectReference
        """
    if described_object is None:
        raise ValueError('Invalid value for `described_object`, must not be `None`')
    self._described_object = described_object