from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_rule_create(self, security_group_id=None, ip_protocol=None, from_port=None, to_port=None, remote_ip=None, remote_group=None):
    """Create a new security group rule

        https://docs.openstack.org/api-ref/compute/#create-security-group-rule

        :param string security_group_id:
            Security group ID
        :param ip_protocol:
            IP protocol, 'tcp', 'udp' or 'icmp'
        :param from_port:
            Source port
        :param to_port:
            Destination port
        :param remote_ip:
            Source IP address in CIDR notation
        :param remote_group:
            Remote security group
        """
    url = '/os-security-group-rules'
    if ip_protocol.lower() not in ['icmp', 'tcp', 'udp']:
        raise InvalidValue("%(s) is not one of 'icmp', 'tcp', or 'udp'" % ip_protocol)
    params = {'parent_group_id': security_group_id, 'ip_protocol': ip_protocol, 'from_port': self._check_integer(from_port), 'to_port': self._check_integer(to_port), 'cidr': remote_ip, 'group_id': remote_group}
    return self.create(url, json={'security_group_rule': params})['security_group_rule']