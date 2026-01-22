import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def change_message_visibility_batch(self, queue, messages):
    """
        A batch version of change_message_visibility that can act
        on up to 10 messages at a time.

        :type queue: A :class:`boto.sqs.queue.Queue` object.
        :param queue: The Queue to which the messages will be written.

        :type messages: List of tuples.
        :param messages: A list of tuples where each tuple consists
            of a :class:`boto.sqs.message.Message` object and an integer
            that represents the new visibility timeout for that message.
        """
    params = {}
    for i, t in enumerate(messages):
        prefix = 'ChangeMessageVisibilityBatchRequestEntry'
        p_name = '%s.%i.Id' % (prefix, i + 1)
        params[p_name] = t[0].id
        p_name = '%s.%i.ReceiptHandle' % (prefix, i + 1)
        params[p_name] = t[0].receipt_handle
        p_name = '%s.%i.VisibilityTimeout' % (prefix, i + 1)
        params[p_name] = t[1]
    return self.get_object('ChangeMessageVisibilityBatch', params, BatchResults, queue.id, verb='POST')