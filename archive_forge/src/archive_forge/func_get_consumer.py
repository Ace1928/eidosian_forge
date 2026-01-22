import abc
import string
from keystone import exception
@abc.abstractmethod
def get_consumer(self, consumer_id):
    """Get consumer, returns the consumer id (key) and description.

        :param consumer_id: id of consumer to get
        :type consumer_id: string
        :returns: consumer_ref

        """
    raise exception.NotImplemented()