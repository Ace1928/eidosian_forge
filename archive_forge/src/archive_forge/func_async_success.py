import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import XmlResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeState
def async_success(self):
    """
        Determinate if async request was successful.

        An async_request retrieves for a task object that can be successfully
        retrieved (self.status == OK), but the asynchronous task (the body of
        the HTTP response) which we are asking for has finished with an error.
        So this method checks if the status code is 'OK' and if the task
        has finished successfully.

        :rtype:  ``bool``
        :return: successful asynchronous request or not
        """
    if self.success():
        task = self.parse_body()
        return task.findtext('state') == 'FINISHED_SUCCESSFULLY'
    else:
        return False