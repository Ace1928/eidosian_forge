from pprint import pformat
from six import iteritems
import re
@disk_name.setter
def disk_name(self, disk_name):
    """
        Sets the disk_name of this V1AzureDiskVolumeSource.
        The Name of the data disk in the blob storage

        :param disk_name: The disk_name of this V1AzureDiskVolumeSource.
        :type: str
        """
    if disk_name is None:
        raise ValueError('Invalid value for `disk_name`, must not be `None`')
    self._disk_name = disk_name