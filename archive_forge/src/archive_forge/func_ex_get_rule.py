from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_get_rule(self, rule_id):
    """
        Get rule by ARN.

        :param rule_id: ARN of rule
        :type rule_id: ``str``

        :return: Rule object
        :rtype: :class:`ALBRule`
        """
    params = {'Action': 'DescribeRules', 'RuleArns.member.1': rule_id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_rules(data)[0]