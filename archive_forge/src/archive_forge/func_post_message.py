from openstack.message.v2 import claim as _claim
from openstack.message.v2 import message as _message
from openstack.message.v2 import queue as _queue
from openstack.message.v2 import subscription as _subscription
from openstack import proxy
from openstack import resource
def post_message(self, queue_name, messages):
    """Post messages to given queue

        :param queue_name: The name of target queue to post message to.
        :param messages: List of messages body and TTL to post.
            :type messages: :py:class:`list`

        :returns: A string includes location of messages successfully posted.
        """
    message = self._get_resource(_message.Message, None, queue_name=queue_name)
    return message.post(self, messages)