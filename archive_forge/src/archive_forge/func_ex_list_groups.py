from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def ex_list_groups(self, location=None):
    """
        List all groups.

        :param location: filter the groups by location (optional)
        :type  location: a :class:`NodeLocation` instance.

        :return:         the list of :class:`NodeGroup`
        """
    groups = []
    for vdc in self._get_locations(location):
        link_vdc = self.connection.cache['locations'][vdc]
        hdr_vdc = {'Accept': self.VDC_MIME_TYPE}
        e_vdc = self.connection.request(link_vdc, headers=hdr_vdc).object
        apps_link = get_href(e_vdc, 'virtualappliances')
        hdr_vapps = {'Accept': self.VAPPS_MIME_TYPE}
        vapps = self.connection.request(apps_link, headers=hdr_vapps).object
        for vapp in vapps.findall('virtualAppliance'):
            nodes = []
            vms_link = get_href(vapp, 'virtualmachines')
            headers = {'Accept': self.NODES_MIME_TYPE}
            vms = self.connection.request(vms_link, headers=headers).object
            for vm in vms.findall('virtualMachine'):
                nodes.append(self._to_node(vm, self))
            group = NodeGroup(self, vapp.findtext('name'), nodes, get_href(vapp, 'edit'))
            groups.append(group)
    return groups