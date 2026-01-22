import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def get_dead_letter_source_queues(self, queue):
    """
        Retrieves the dead letter source queues for a given queue.

        :type queue: A :class:`boto.sqs.queue.Queue` object.
        :param queue: The queue for which to get DL source queues
        :rtype: list
        :returns: A list of :py:class:`boto.sqs.queue.Queue` instances.
        """
    params = {'QueueUrl': queue.url}
    return self.get_list('ListDeadLetterSourceQueues', params, [('QueueUrl', Queue)])