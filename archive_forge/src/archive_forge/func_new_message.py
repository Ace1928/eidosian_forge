from boto.compat import urllib
from boto.sqs.message import Message
def new_message(self, body='', **kwargs):
    """
        Create new message of appropriate class.

        :type body: message body
        :param body: The body of the newly created message (optional).

        :rtype: :class:`boto.sqs.message.Message`
        :return: A new Message object
        """
    m = self.message_class(self, body, **kwargs)
    m.queue = self
    return m