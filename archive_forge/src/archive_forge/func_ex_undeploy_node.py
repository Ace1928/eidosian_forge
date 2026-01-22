import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_undeploy_node(self, node, shutdown=True):
    """
        Undeploys existing node. Equal to vApp "stop" operation.

        :param  node: The node to be deployed
        :type   node: :class:`Node`

        :param  shutdown: Whether to shutdown or power off the guest when
                undeploying
        :type   shutdown: ``bool``

        :rtype: :class:`Node`
        """
    data = {'xmlns': 'http://www.vmware.com/vcloud/v1.5'}
    undeploy_xml = ET.Element('UndeployVAppParams', data)
    undeploy_power_action_xml = ET.SubElement(undeploy_xml, 'UndeployPowerAction')
    headers = {'Content-Type': 'application/vnd.vmware.vcloud.undeployVAppParams+xml'}

    def undeploy(action):
        undeploy_power_action_xml.text = action
        undeploy_res = self.connection.request('%s/action/undeploy' % get_url_path(node.id), data=ET.tostring(undeploy_xml), method='POST', headers=headers)
        self._wait_for_task_completion(undeploy_res.object.get('href'))
    if shutdown:
        try:
            undeploy('shutdown')
        except Exception:
            undeploy('powerOff')
    else:
        undeploy('powerOff')
    res = self.connection.request(get_url_path(node.id))
    return self._to_node(res.object)