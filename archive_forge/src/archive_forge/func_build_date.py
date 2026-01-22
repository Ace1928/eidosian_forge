from pprint import pformat
from six import iteritems
import re
@build_date.setter
def build_date(self, build_date):
    """
        Sets the build_date of this VersionInfo.

        :param build_date: The build_date of this VersionInfo.
        :type: str
        """
    if build_date is None:
        raise ValueError('Invalid value for `build_date`, must not be `None`')
    self._build_date = build_date