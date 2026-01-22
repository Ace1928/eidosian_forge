import abc
from keystone import exception
@abc.abstractmethod
def create_policy_association(self, policy_id, endpoint_id=None, service_id=None, region_id=None):
    """Create a policy association.

        :param policy_id: identity of policy that is being associated
        :type policy_id: string
        :param endpoint_id: identity of endpoint to associate
        :type endpoint_id: string
        :param service_id: identity of the service to associate
        :type service_id: string
        :param region_id: identity of the region to associate
        :type region_id: string
        :returns: None

        There are three types of association permitted:

        - Endpoint (in which case service and region must be None)
        - Service and region (in which endpoint must be None)
        - Service (in which case endpoint and region must be None)

        """
    raise exception.NotImplemented()