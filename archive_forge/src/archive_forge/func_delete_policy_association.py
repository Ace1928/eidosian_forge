import abc
from keystone import exception
@abc.abstractmethod
def delete_policy_association(self, policy_id, endpoint_id=None, service_id=None, region_id=None):
    """Delete a policy association.

        :param policy_id: identity of policy that is being associated
        :type policy_id: string
        :param endpoint_id: identity of endpoint to associate
        :type endpoint_id: string
        :param service_id: identity of the service to associate
        :type service_id: string
        :param region_id: identity of the region to associate
        :type region_id: string
        :returns: None

        """
    raise exception.NotImplemented()