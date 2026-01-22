from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_get_target_group(self, target_group_id):
    """
        Get target group object by ARN

        :param target_group_id: ARN of target group
        :type target_group_id: ``str``

        :return: Target group object
        :rtype: :class:`ALBTargetGroup`
        """
    params = {'Action': 'DescribeTargetGroups', 'TargetGroupArns.member.1': target_group_id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_target_groups(data)[0]