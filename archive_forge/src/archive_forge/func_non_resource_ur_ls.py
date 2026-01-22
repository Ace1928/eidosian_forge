from pprint import pformat
from six import iteritems
import re
@non_resource_ur_ls.setter
def non_resource_ur_ls(self, non_resource_ur_ls):
    """
        Sets the non_resource_ur_ls of this V1beta1NonResourceRule.
        NonResourceURLs is a set of partial urls that a user should have access
        to.  *s are allowed, but only as the full, final step in the path.
        "*" means all.

        :param non_resource_ur_ls: The non_resource_ur_ls of this
        V1beta1NonResourceRule.
        :type: list[str]
        """
    self._non_resource_ur_ls = non_resource_ur_ls