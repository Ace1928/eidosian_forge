from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def get_vnic_metrics(self, vm_name):
    ports = self._get_vm_resources(vm_name, self._PORT_ALLOC_SET_DATA)
    vnics = self._get_vm_resources(vm_name, self._SYNTH_ETH_PORT_SET_DATA)
    metrics_def_in = self._metrics_defs[self._NET_IN_METRICS]
    metrics_def_out = self._metrics_defs[self._NET_OUT_METRICS]
    for port in ports:
        vnic = [v for v in vnics if port.Parent == v.path_()][0]
        port_acls = _wqlutils.get_element_associated_class(self._conn, self._PORT_ALLOC_ACL_SET_DATA, element_instance_id=port.InstanceID)
        metrics_value_instances = self._get_metrics_value_instances(port_acls, self._BASE_METRICS_VALUE)
        metrics_values = self._sum_metrics_values_by_defs(metrics_value_instances, [metrics_def_in, metrics_def_out])
        yield {'rx_mb': metrics_values[0], 'tx_mb': metrics_values[1], 'element_name': vnic.ElementName, 'address': vnic.Address}