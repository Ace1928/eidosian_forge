from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def resolve_ext_resource(self, resource, name_or_id):
    """Returns the id and validate neutron ext resource."""
    path = self._resolve_resource_path(resource)
    try:
        record = self.client().show_ext(path + '/%s', name_or_id)
        return record.get(resource).get('id')
    except exceptions.NotFound:
        res_plural = resource + 's'
        result = self.client().list_ext(collection=res_plural, path=path, retrieve_all=True)
        resources = result.get(res_plural)
        matched = []
        for res in resources:
            if res.get('name') == name_or_id:
                matched.append(res.get('id'))
        if len(matched) > 1:
            raise exceptions.NeutronClientNoUniqueMatch(resource=resource, name=name_or_id)
        elif len(matched) == 0:
            not_found_message = _("Unable to find %(resource)s with name or id '%(name_or_id)s'") % {'resource': resource, 'name_or_id': name_or_id}
            raise exceptions.NotFound(message=not_found_message)
        else:
            return matched[0]