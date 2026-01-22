from pprint import pformat
from six import iteritems
import re
@failed_jobs_history_limit.setter
def failed_jobs_history_limit(self, failed_jobs_history_limit):
    """
        Sets the failed_jobs_history_limit of this V2alpha1CronJobSpec.
        The number of failed finished jobs to retain. This is a pointer to
        distinguish between explicit zero and not specified.

        :param failed_jobs_history_limit: The failed_jobs_history_limit of this
        V2alpha1CronJobSpec.
        :type: int
        """
    self._failed_jobs_history_limit = failed_jobs_history_limit