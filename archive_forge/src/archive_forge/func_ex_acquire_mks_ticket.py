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
def ex_acquire_mks_ticket(self, vapp_or_vm_id, vm_num=0):
    """
        Retrieve a mks ticket that you can use to gain access to the console
        of a running VM. If successful, returns a dict with the following
        keys:

          - host: host (or proxy) through which the console connection
                is made
          - vmx: a reference to the VMX file of the VM for which this
               ticket was issued
          - ticket: screen ticket to use to authenticate the client
          - port: host port to be used for console access

        :param  vapp_or_vm_id: vApp or VM ID you want to connect to.
        :type   vapp_or_vm_id: ``str``

        :param  vm_num: If a vApp ID is provided, vm_num is position in the
                vApp VM list of the VM you want to get a screen ticket.
                Default is 0.
        :type   vm_num: ``int``

        :rtype: ``dict``
        """
    vm = self._get_vm_elements(vapp_or_vm_id)[vm_num]
    try:
        res = self.connection.request('%s/screen/action/acquireMksTicket' % get_url_path(vm.get('href')), method='POST')
        output = {'host': res.object.find(fixxpath(res.object, 'Host')).text, 'vmx': res.object.find(fixxpath(res.object, 'Vmx')).text, 'ticket': res.object.find(fixxpath(res.object, 'Ticket')).text, 'port': res.object.find(fixxpath(res.object, 'Port')).text}
        return output
    except Exception:
        return None