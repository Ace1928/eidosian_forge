from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def get_all_lb_attributes(self, load_balancer_name):
    """Gets all Attributes of a Load Balancer

        :type load_balancer_name: string
        :param load_balancer_name: The name of the Load Balancer

        :rtype: boto.ec2.elb.attribute.LbAttributes
        :return: The attribute object of the ELB.
        """
    from boto.ec2.elb.attributes import LbAttributes
    params = {'LoadBalancerName': load_balancer_name}
    return self.get_object('DescribeLoadBalancerAttributes', params, LbAttributes)