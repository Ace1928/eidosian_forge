from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def set_lb_policies_of_backend_server(self, lb_name, instance_port, policies):
    """
        Replaces the current set of policies associated with a port on which
        the back-end server is listening with a new set of policies.
        """
    params = {'LoadBalancerName': lb_name, 'InstancePort': instance_port}
    if policies:
        self.build_list_params(params, policies, 'PolicyNames.member.%d')
    else:
        params['PolicyNames'] = ''
    return self.get_status('SetLoadBalancerPoliciesForBackendServer', params)