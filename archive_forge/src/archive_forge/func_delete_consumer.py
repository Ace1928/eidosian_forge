import abc
import string
from keystone import exception
@abc.abstractmethod
def delete_consumer(self, consumer_id):
    """Delete consumer.

        :param consumer_id: id of consumer to get
        :type consumer_id: string
        :returns: None.

        """
    raise exception.NotImplemented()