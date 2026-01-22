from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_set_balancer_policies_backend_server(self, name, instance_port, policies):
    """
        Replaces the current set of policies associated with a port on
        which the back-end server is listening with a new set of policies

        :param name: balancer name to set policies of backend server
        :type  name: ``str``

        :param instance_port: Instance Port
        :type  instance_port: ``int``

        :param policies: List of policies to be associated with the balancer
        :type  policies: ``string list`
        """
    params = {'Action': 'SetLoadBalancerPoliciesForBackendServer', 'LoadBalancerName': name, 'InstancePort': str(instance_port)}
    if policies:
        params = self._create_list_params(params, policies, 'PolicyNames.member.%d')
    response = self.connection.request(ROOT, params=params)
    return response.status == httplib.OK