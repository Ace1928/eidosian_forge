from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_update_listener(self, virtual_listener, **kwargs):
    """
        Update a current virtual listener.
        :param virtual_listener: The listener to be updated
        :return: The edited version of the listener
        """
    edit_listener_elm = ET.Element('editVirtualListener', {'xmlns': TYPES_URN, 'id': virtual_listener.id, 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
    for k, v in kwargs.items():
        if v is None:
            ET.SubElement(edit_listener_elm, k, {'xsi:nil': 'true'})
        else:
            ET.SubElement(edit_listener_elm, k).text = v
    result = self.connection.request_with_orgId_api_2('networkDomainVip/editVirtualListener', method='POST', data=ET.tostring(edit_listener_elm)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']