from pprint import pformat
from six import iteritems
import re
@delete_options.setter
def delete_options(self, delete_options):
    """
        Sets the delete_options of this V1beta1Eviction.
        DeleteOptions may be provided

        :param delete_options: The delete_options of this V1beta1Eviction.
        :type: V1DeleteOptions
        """
    self._delete_options = delete_options