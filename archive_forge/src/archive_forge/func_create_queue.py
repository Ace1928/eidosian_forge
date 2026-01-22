import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def create_queue(self, queue_name, visibility_timeout=None):
    """
        Create an SQS Queue.

        :type queue_name: str or unicode
        :param queue_name: The name of the new queue.  Names are
            scoped to an account and need to be unique within that
            account.  Calling this method on an existing queue name
            will not return an error from SQS unless the value for
            visibility_timeout is different than the value of the
            existing queue of that name.  This is still an expensive
            operation, though, and not the preferred way to check for
            the existence of a queue.  See the
            :func:`boto.sqs.connection.SQSConnection.lookup` method.

        :type visibility_timeout: int
        :param visibility_timeout: The default visibility timeout for
            all messages written in the queue.  This can be overridden
            on a per-message.

        :rtype: :class:`boto.sqs.queue.Queue`
        :return: The newly created queue.

        """
    params = {'QueueName': queue_name}
    if visibility_timeout:
        params['Attribute.1.Name'] = 'VisibilityTimeout'
        params['Attribute.1.Value'] = int(visibility_timeout)
    return self.get_object('CreateQueue', params, Queue)