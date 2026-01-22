from pprint import pformat
from six import iteritems
import re
@backoff_limit.setter
def backoff_limit(self, backoff_limit):
    """
        Sets the backoff_limit of this V1JobSpec.
        Specifies the number of retries before marking this job failed. Defaults
        to 6

        :param backoff_limit: The backoff_limit of this V1JobSpec.
        :type: int
        """
    self._backoff_limit = backoff_limit