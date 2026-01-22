import abc
from keystone import exception
@abc.abstractmethod
def delete_association_by_endpoint(self, endpoint_id):
    """Remove all the policy associations with the specific endpoint.

        :param endpoint_id: identity of endpoint to check
        :type endpoint_id: string
        :returns: None

        """
    raise exception.NotImplemented()