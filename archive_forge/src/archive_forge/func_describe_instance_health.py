from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def describe_instance_health(self, load_balancer_name, instances=None):
    """
        Get current state of all Instances registered to an Load Balancer.

        :type load_balancer_name: string
        :param load_balancer_name: The name of the Load Balancer

        :type instances: List of strings
        :param instances: The instance ID's of the EC2 instances
                          to return status for.  If not provided,
                          the state of all instances will be returned.

        :rtype: List of :class:`boto.ec2.elb.instancestate.InstanceState`
        :return: list of state info for instances in this Load Balancer.

        """
    params = {'LoadBalancerName': load_balancer_name}
    if instances:
        self.build_list_params(params, instances, 'Instances.member.%d.InstanceId')
    return self.get_list('DescribeInstanceHealth', params, [('member', InstanceState)])