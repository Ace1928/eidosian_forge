from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def connect_to_region(region_name, **kw_params):
    """
    Given a valid region name, return a
    :class:`boto.ec2.elb.ELBConnection`.

    :param str region_name: The name of the region to connect to.

    :rtype: :class:`boto.ec2.ELBConnection` or ``None``
    :return: A connection to the given region, or None if an invalid region
        name is given
    """
    return connect('elasticloadbalancing', region_name, connection_cls=ELBConnection, **kw_params)