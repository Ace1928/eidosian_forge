from pprint import pformat
from six import iteritems
import re
@share_name.setter
def share_name(self, share_name):
    """
        Sets the share_name of this V1AzureFileVolumeSource.
        Share Name

        :param share_name: The share_name of this V1AzureFileVolumeSource.
        :type: str
        """
    if share_name is None:
        raise ValueError('Invalid value for `share_name`, must not be `None`')
    self._share_name = share_name