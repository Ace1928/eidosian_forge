from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def create_lb_cookie_stickiness_policy(self, cookie_expiration_period, lb_name, policy_name):
    """
        Generates a stickiness policy with sticky session lifetimes controlled
        by the lifetime of the browser (user-agent) or a specified expiration
        period. This policy can only be associated only with HTTP listeners.

        When a load balancer implements this policy, the load balancer uses a
        special cookie to track the backend server instance for each request.
        When the load balancer receives a request, it first checks to see if
        this cookie is present in the request. If so, the load balancer sends
        the request to the application server specified in the cookie. If not,
        the load balancer sends the request to a server that is chosen based on
        the existing load balancing algorithm.

        A cookie is inserted into the response for binding subsequent requests
        from the same user to that server. The validity of the cookie is based
        on the cookie expiration time, which is specified in the policy
        configuration.

        None may be passed for cookie_expiration_period.
        """
    params = {'LoadBalancerName': lb_name, 'PolicyName': policy_name}
    if cookie_expiration_period is not None:
        params['CookieExpirationPeriod'] = cookie_expiration_period
    return self.get_status('CreateLBCookieStickinessPolicy', params)