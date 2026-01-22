import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def ex_node_action(self, node, action):
    """
        Build action representation and instruct node to commit action.

        Build action representation from the compute node ID, and the
        action which should be carried out on that compute node. Then
        instruct the node to carry out that action.

        :param node: Compute node instance.
        :type  node: :class:`Node`

        :param action: Action to be carried out on the compute node.
        :type  action: ``str``

        :return: False if an HTTP Bad Request is received, else, True is
                 returned.
        :rtype:  ``bool``
        """
    compute_node_id = str(node.id)
    compute = ET.Element('COMPUTE')
    compute_id = ET.SubElement(compute, 'ID')
    compute_id.text = compute_node_id
    state = ET.SubElement(compute, 'STATE')
    state.text = action
    xml = ET.tostring(compute)
    url = '/compute/%s' % compute_node_id
    resp = self.connection.request(url, method='PUT', data=xml)
    if resp.status == httplib.BAD_REQUEST:
        return False
    else:
        return True