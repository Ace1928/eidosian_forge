import logging
from oslo_vmware import vim_util
def get_trunk_vlan_spec(session, start=0, end=4094):
    """Gets portgroup trunk vlan spec.

    :param session: vCenter soap session
    :param start: the starting id
    :param end: then end id
    :returns: The configuration when a port uses trunk mode. This allows
              a guest to manage the vlan id.
    """
    client_factory = session.vim.client.factory
    spec_ns = 'ns0:VmwareDistributedVirtualSwitchTrunkVlanSpec'
    vlan_id = client_factory.create('ns0:NumericRange')
    vlan_id.start = start
    vlan_id.end = end
    vl_spec = client_factory.create(spec_ns)
    vl_spec.vlanId = vlan_id
    vl_spec.inherited = '0'
    return vl_spec