import logging
from botocore import xform_name
@property
def subresources(self):
    """
        Get a list of sub-resources.

        :type: list(:py:class:`ResponseResource`)
        """
    return self._get_related_resources(True)