import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def set_port_qos_rule(self, port_id, qos_rule):
    """Sets the QoS rule for the given port.

        :param port_id: the port's ID to which the QoS rule will be applied to.
        :param qos_rule: a dictionary containing the following keys:
            min_kbps, max_kbps, max_burst_kbps, max_burst_size_kb.
        :raises exceptions.HyperVInvalidException: if
            - min_kbps is smaller than 10MB.
            - max_kbps is smaller than min_kbps.
            - max_burst_kbps is smaller than max_kbps.
        :raises exceptions.HyperVException: if the QoS rule cannot be set.
        """
    min_bps = qos_rule.get('min_kbps', 0) * units.Ki
    max_bps = qos_rule.get('max_kbps', 0) * units.Ki
    max_burst_bps = qos_rule.get('max_burst_kbps', 0) * units.Ki
    max_burst_sz = qos_rule.get('max_burst_size_kb', 0) * units.Ki
    if not (min_bps or max_bps or max_burst_bps or max_burst_sz):
        return
    if min_bps and min_bps < 10 * units.Mi:
        raise exceptions.InvalidParameterValue(param_name='min_kbps', param_value=min_bps)
    if max_bps and max_bps < min_bps:
        raise exceptions.InvalidParameterValue(param_name='max_kbps', param_value=max_bps)
    if max_burst_bps and max_burst_bps < max_bps:
        raise exceptions.InvalidParameterValue(param_name='max_burst_kbps', param_value=max_burst_bps)
    port_alloc = self._get_switch_port_allocation(port_id)[0]
    bandwidth = self._get_bandwidth_setting_data_from_port_alloc(port_alloc)
    if bandwidth:
        self._jobutils.remove_virt_feature(bandwidth)
        self._bandwidth_sds.pop(port_alloc.InstanceID, None)
    bandwidth = self._get_default_setting_data(self._PORT_BANDWIDTH_SET_DATA)
    bandwidth.Reservation = min_bps
    bandwidth.Limit = max_bps
    bandwidth.BurstLimit = max_burst_bps
    bandwidth.BurstSize = max_burst_sz
    try:
        self._jobutils.add_virt_feature(bandwidth, port_alloc)
    except Exception as ex:
        if '0x80070057' in six.text_type(ex):
            raise exceptions.InvalidParameterValue(param_name='qos_rule', param_value=qos_rule)
        raise exceptions.HyperVException('Unable to set qos rule %(qos_rule)s for port %(port)s. Error: %(error)s' % dict(qos_rule=qos_rule, port=port_alloc, error=ex))