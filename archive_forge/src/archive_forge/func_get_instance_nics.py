from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def get_instance_nics(self, instance):
    nics = []
    for nic in instance[self.NETWORKS]:
        nic_dict = {}
        if nic.get(self.NET):
            nic_dict['net-id'] = nic.get(self.NET)
        if nic.get(self.PORT):
            nic_dict['port-id'] = nic.get(self.PORT)
        ip = nic.get(self.V4_FIXED_IP)
        if ip:
            nic_dict['v4-fixed-ip'] = ip
        nics.append(nic_dict)
    return nics