from pprint import pformat
from six import iteritems
import re
@starting_deadline_seconds.setter
def starting_deadline_seconds(self, starting_deadline_seconds):
    """
        Sets the starting_deadline_seconds of this V2alpha1CronJobSpec.
        Optional deadline in seconds for starting the job if it misses scheduled
        time for any reason.  Missed jobs executions will be counted as failed
        ones.

        :param starting_deadline_seconds: The starting_deadline_seconds of this
        V2alpha1CronJobSpec.
        :type: int
        """
    self._starting_deadline_seconds = starting_deadline_seconds