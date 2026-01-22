from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def detach_lb_from_subnets(self, name, subnets):
    """
        Detaches load balancer from one or more subnets.

        :type name: string
        :param name: The name of the Load Balancer

        :type subnets: List of strings
        :param subnets: The name of the subnet(s) to detach.

        :rtype: List of strings
        :return: An updated list of subnets for this Load Balancer.

        """
    params = {'LoadBalancerName': name}
    self.build_list_params(params, subnets, 'Subnets.member.%d')
    return self.get_list('DetachLoadBalancerFromSubnets', params, None)