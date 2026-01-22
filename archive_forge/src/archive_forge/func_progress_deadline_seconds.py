from pprint import pformat
from six import iteritems
import re
@progress_deadline_seconds.setter
def progress_deadline_seconds(self, progress_deadline_seconds):
    """
        Sets the progress_deadline_seconds of this V1beta2DeploymentSpec.
        The maximum time in seconds for a deployment to make progress before it
        is considered to be failed. The deployment controller will continue to
        process failed deployments and a condition with a
        ProgressDeadlineExceeded reason will be surfaced in the deployment
        status. Note that progress will not be estimated during the time a
        deployment is paused. Defaults to 600s.

        :param progress_deadline_seconds: The progress_deadline_seconds of this
        V1beta2DeploymentSpec.
        :type: int
        """
    self._progress_deadline_seconds = progress_deadline_seconds