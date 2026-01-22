from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def ex_create_group(self, name, location=None):
    """
        Create an empty group.

        You can specify the location as well.

        :param     group:     name of the group (required)
        :type      group:     ``str``

        :param     location: location were to create the group
        :type      location: :class:`NodeLocation`

        :returns:            the created group
        :rtype:              :class:`NodeGroup`
        """
    vapp = ET.Element('virtualAppliance')
    vapp_name = ET.SubElement(vapp, 'name')
    vapp_name.text = name
    if location is None:
        location = self.list_locations()[0]
    elif location not in self.list_locations():
        raise LibcloudError('Location does not exist')
    link_vdc = self.connection.cache['locations'][location]
    hdr_vdc = {'Accept': self.VDC_MIME_TYPE}
    e_vdc = self.connection.request(link_vdc, headers=hdr_vdc).object
    creation_link = get_href(e_vdc, 'virtualappliances')
    headers = {'Accept': self.VAPP_MIME_TYPE, 'Content-type': self.VAPP_MIME_TYPE}
    vapp = self.connection.request(creation_link, data=tostring(vapp), headers=headers, method='POST').object
    uri_vapp = get_href(vapp, 'edit')
    return NodeGroup(self, vapp.findtext('name'), uri=uri_vapp)