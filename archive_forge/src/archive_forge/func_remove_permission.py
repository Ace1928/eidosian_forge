import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def remove_permission(self, queue, label):
    """
        Remove a permission from a queue.

        :type queue: :class:`boto.sqs.queue.Queue`
        :param queue: The queue object

        :type label: str or unicode
        :param label: The unique label associated with the permission
                      being removed.

        :rtype: bool
        :return: True if successful, False otherwise.
        """
    params = {'Label': label}
    return self.get_status('RemovePermission', params, queue.id)