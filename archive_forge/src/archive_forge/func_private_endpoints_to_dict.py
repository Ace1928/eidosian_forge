from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def private_endpoints_to_dict(self, privateendpoint):
    results = dict(id=privateendpoint.id, name=privateendpoint.name, location=privateendpoint.location, tags=privateendpoint.tags, provisioning_state=privateendpoint.provisioning_state, type=privateendpoint.type, etag=privateendpoint.etag, subnet=dict(id=privateendpoint.subnet.id))
    if privateendpoint.network_interfaces and len(privateendpoint.network_interfaces) > 0:
        results['network_interfaces'] = []
        for interface in privateendpoint.network_interfaces:
            results['network_interfaces'].append(interface.id)
    if privateendpoint.private_link_service_connections and len(privateendpoint.private_link_service_connections) > 0:
        results['private_link_service_connections'] = []
        for connections in privateendpoint.private_link_service_connections:
            results['private_link_service_connections'].append(dict(private_link_service_id=connections.private_link_service_id, name=connections.name))
    return results