from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def create_load_balancer_listeners(self, name, listeners=None, complex_listeners=None):
    """
        Creates a Listener (or group of listeners) for an existing
        Load Balancer

        :type name: string
        :param name: The name of the load balancer to create the listeners for

        :type listeners: List of tuples
        :param listeners: Each tuple contains three or four values,
            (LoadBalancerPortNumber, InstancePortNumber, Protocol,
            [SSLCertificateId]) where LoadBalancerPortNumber and
            InstancePortNumber are integer values between 1 and 65535,
            Protocol is a string containing either 'TCP', 'SSL', HTTP', or
            'HTTPS'; SSLCertificateID is the ARN of a AWS IAM
            certificate, and must be specified when doing HTTPS.

        :type complex_listeners: List of tuples
        :param complex_listeners: Each tuple contains four or five values,
            (LoadBalancerPortNumber, InstancePortNumber, Protocol,
             InstanceProtocol, SSLCertificateId).

            Where:
                - LoadBalancerPortNumber and InstancePortNumber are integer
                  values between 1 and 65535
                - Protocol and InstanceProtocol is a string containing
                  either 'TCP',
                  'SSL', 'HTTP', or 'HTTPS'
                - SSLCertificateId is the ARN of an SSL certificate loaded into
                  AWS IAM

        :return: The status of the request
        """
    if not listeners and (not complex_listeners):
        return None
    params = {'LoadBalancerName': name}
    if listeners:
        for index, listener in enumerate(listeners):
            i = index + 1
            protocol = listener[2].upper()
            params['Listeners.member.%d.LoadBalancerPort' % i] = listener[0]
            params['Listeners.member.%d.InstancePort' % i] = listener[1]
            params['Listeners.member.%d.Protocol' % i] = listener[2]
            if protocol == 'HTTPS' or protocol == 'SSL':
                params['Listeners.member.%d.SSLCertificateId' % i] = listener[3]
    if complex_listeners:
        for index, listener in enumerate(complex_listeners):
            i = index + 1
            protocol = listener[2].upper()
            InstanceProtocol = listener[3].upper()
            params['Listeners.member.%d.LoadBalancerPort' % i] = listener[0]
            params['Listeners.member.%d.InstancePort' % i] = listener[1]
            params['Listeners.member.%d.Protocol' % i] = listener[2]
            params['Listeners.member.%d.InstanceProtocol' % i] = listener[3]
            if protocol == 'HTTPS' or protocol == 'SSL':
                params['Listeners.member.%d.SSLCertificateId' % i] = listener[4]
    return self.get_status('CreateLoadBalancerListeners', params)