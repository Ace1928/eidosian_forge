from pprint import pformat
from six import iteritems
import re
@singular_name.setter
def singular_name(self, singular_name):
    """
        Sets the singular_name of this V1APIResource.
        singularName is the singular name of the resource.  This allows clients
        to handle plural and singular opaquely. The singularName is more correct
        for reporting status on a single item and both singular and plural are
        allowed from the kubectl CLI interface.

        :param singular_name: The singular_name of this V1APIResource.
        :type: str
        """
    if singular_name is None:
        raise ValueError('Invalid value for `singular_name`, must not be `None`')
    self._singular_name = singular_name