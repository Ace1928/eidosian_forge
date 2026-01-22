from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_create_balancer_policy(self, name, policy_name, policy_type, policy_attributes=None):
    """
        Create a new load balancer policy

        :param name: Balancer name to create the policy for
        :type  name: ``str``

        :param policy_name: policy to be created
        :type  policy_name: ``str``

        :param policy_type: policy type being used to create policy.
        :type  policy_type: ``str``

        :param policy_attributes: Each list contain values, ['AttributeName',
                                                             'value']
        :type  policy_attributes: ``PolicyAttribute list``
        """
    params = {'Action': 'CreateLoadBalancerPolicy', 'LoadBalancerName': name, 'PolicyName': policy_name, 'PolicyTypeName': policy_type}
    if policy_attributes is not None:
        for index, (name, value) in enumerate(policy_attributes.iteritems(), 1):
            params['PolicyAttributes.member.%d.                         AttributeName' % index] = name
            params['PolicyAttributes.member.%d.                         AttributeValue' % index] = value
    response = self.connection.request(ROOT, params=params)
    return response.status == httplib.OK