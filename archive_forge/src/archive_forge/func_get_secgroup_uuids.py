from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def get_secgroup_uuids(self, security_groups):
    """Returns a list of security group UUIDs.

        Args:
        security_groups: List of security group names or UUIDs
        """
    seclist = []
    all_groups = None
    for sg in security_groups:
        if uuidutils.is_uuid_like(sg):
            seclist.append(sg)
        else:
            if not all_groups:
                response = self.client().list_security_groups(project_id=self.context.project_id)
                all_groups = response['security_groups']
            same_name_groups = [g for g in all_groups if g['name'] == sg]
            groups = [g['id'] for g in same_name_groups]
            if len(groups) == 0:
                raise exception.EntityNotFound(entity='Resource', name=sg)
            elif len(groups) == 1:
                seclist.append(groups[0])
            else:
                raise exception.PhysicalResourceNameAmbiguity(name=sg)
    return seclist