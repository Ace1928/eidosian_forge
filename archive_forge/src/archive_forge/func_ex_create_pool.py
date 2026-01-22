from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_create_pool(self, network_domain_id, name, balancer_method, ex_description, health_monitors=None, service_down_action='NONE', slow_ramp_time=30):
    """
        Create a new pool

        :param network_domain_id: Network Domain ID (required)
        :type  name: ``str``

        :param name: name of the node (required)
        :type  name: ``str``

        :param balancer_method: The load balancer algorithm (required)
        :type  balancer_method: ``str``

        :param ex_description: Description of the node (required)
        :type  ex_description: ``str``

        :param health_monitors: A list of health monitors to use for the pool.
        :type  health_monitors: ``list`` of
            :class:`NttCisDefaultHealthMonitor`

        :param service_down_action: What to do when node
                                    is unavailable NONE, DROP or RESELECT
        :type  service_down_action: ``str``

        :param slow_ramp_time: Number of seconds to stagger ramp up of nodes
        :type  slow_ramp_time: ``int``

        :return: Instance of ``NttCisPool``
        :rtype: ``NttCisPool``
        """
    name.replace(' ', '_')
    create_node_elm = ET.Element('createPool', {'xmlns': TYPES_URN})
    ET.SubElement(create_node_elm, 'networkDomainId').text = network_domain_id
    ET.SubElement(create_node_elm, 'name').text = name
    ET.SubElement(create_node_elm, 'description').text = str(ex_description)
    ET.SubElement(create_node_elm, 'loadBalanceMethod').text = str(balancer_method)
    if health_monitors is not None:
        for monitor in health_monitors:
            ET.SubElement(create_node_elm, 'healthMonitorId').text = str(monitor.id)
    ET.SubElement(create_node_elm, 'serviceDownAction').text = service_down_action
    ET.SubElement(create_node_elm, 'slowRampTime').text = str(slow_ramp_time)
    response = self.connection.request_with_orgId_api_2(action='networkDomainVip/createPool', method='POST', data=ET.tostring(create_node_elm)).object
    pool_id = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'poolId':
            pool_id = info.get('value')
    return NttCisPool(id=pool_id, name=name, description=ex_description, status=State.RUNNING, load_balance_method=str(balancer_method), health_monitor_id=None, service_down_action=service_down_action, slow_ramp_time=str(slow_ramp_time))