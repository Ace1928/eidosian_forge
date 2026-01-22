import collections
import copy
from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_security_group_neutron_update(self):
    self.stubout_neutron_create_security_group(mock_csgr=False)
    self.stubout_neutron_get_security_group()
    stack = self.create_stack(self.test_template_neutron)
    sg = stack['the_sg']
    self.assertResourceState(sg, 'aaaa')
    props = copy.deepcopy(sg.properties.data)
    props['SecurityGroupIngress'] = [{'IpProtocol': 'tcp', 'FromPort': '80', 'ToPort': '80', 'CidrIp': '0.0.0.0/0'}, {'IpProtocol': 'tcp', 'FromPort': '443', 'ToPort': '443', 'CidrIp': '0.0.0.0/0'}, {'IpProtocol': 'tcp', 'SourceSecurityGroupId': 'zzzz'}]
    props['SecurityGroupEgress'] = [{'IpProtocol': 'tcp', 'FromPort': '22', 'ToPort': '22', 'CidrIp': '0.0.0.0/0'}, {'SourceSecurityGroupName': 'xxxx'}]
    after = rsrc_defn.ResourceDefinition(sg.name, sg.type(), props)
    self.m_csgr.side_effect = [{'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 443, 'ethertype': 'IPv4', 'port_range_max': 443, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'bbbb'}}, {'security_group_rule': {'direction': 'ingress', 'remote_group_id': 'zzzz', 'remote_ip_prefix': None, 'port_range_min': None, 'ethertype': 'IPv4', 'port_range_max': None, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'dddd'}}, {'security_group_rule': {'direction': 'egress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'eeee'}}]
    scheduler.TaskRunner(sg.update, after)()
    self.assertEqual((sg.UPDATE, sg.COMPLETE), sg.state)
    self.m_dsgr.assert_has_calls([mock.call('aaaa-1'), mock.call('aaaa-2'), mock.call('eeee'), mock.call('dddd'), mock.call('bbbb')], any_order=True)
    self.m_ssg.assert_called_once_with('aaaa')