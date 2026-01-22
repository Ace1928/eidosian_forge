from oslo_log import log
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def list_endpoints_for_policy(self, policy_id):

    def _get_endpoint(endpoint_id, policy_id):
        try:
            return PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        except exception.EndpointNotFound:
            msg = 'Endpoint %(endpoint_id)s referenced in association for policy %(policy_id)s not found.'
            LOG.warning(msg, {'policy_id': policy_id, 'endpoint_id': endpoint_id})
            raise

    def _get_endpoints_for_service(service_id, endpoints):
        return [ep for ep in endpoints if ep['service_id'] == service_id]

    def _get_endpoints_for_service_and_region(service_id, region_id, endpoints, regions):

        def _recursively_get_endpoints_for_region(region_id, service_id, endpoint_list, region_list, endpoints_found, regions_examined):
            """Recursively search down a region tree for endpoints.

                :param region_id: the point in the tree to examine
                :param service_id: the service we are interested in
                :param endpoint_list: list of all endpoints
                :param region_list: list of all regions
                :param endpoints_found: list of matching endpoints found so
                                        far - which will be updated if more are
                                        found in this iteration
                :param regions_examined: list of regions we have already looked
                                         at - used to spot illegal circular
                                         references in the tree to avoid never
                                         completing search
                :returns: list of endpoints that match

                """
            if region_id in regions_examined:
                msg = 'Circular reference or a repeated entry found in region tree - %(region_id)s.'
                LOG.error(msg, {'region_id': ref.region_id})
                return
            regions_examined.append(region_id)
            endpoints_found += [ep for ep in endpoint_list if ep['service_id'] == service_id and ep['region_id'] == region_id]
            for region in region_list:
                if region['parent_region_id'] == region_id:
                    _recursively_get_endpoints_for_region(region['id'], service_id, endpoints, regions, endpoints_found, regions_examined)
        endpoints_found = []
        regions_examined = []
        _recursively_get_endpoints_for_region(region_id, service_id, endpoints, regions, endpoints_found, regions_examined)
        return endpoints_found
    matching_endpoints = []
    endpoints = PROVIDERS.catalog_api.list_endpoints()
    regions = PROVIDERS.catalog_api.list_regions()
    for ref in self.list_associations_for_policy(policy_id):
        if ref.get('endpoint_id') is not None:
            matching_endpoints.append(_get_endpoint(ref['endpoint_id'], policy_id))
            continue
        if ref.get('service_id') is not None and ref.get('region_id') is None:
            matching_endpoints += _get_endpoints_for_service(ref['service_id'], endpoints)
            continue
        if ref.get('service_id') is not None and ref.get('region_id') is not None:
            matching_endpoints += _get_endpoints_for_service_and_region(ref['service_id'], ref['region_id'], endpoints, regions)
            continue
        msg = 'Unsupported policy association found - Policy %(policy_id)s, Endpoint %(endpoint_id)s, Service %(service_id)s, Region %(region_id)s, '
        LOG.warning(msg, {'policy_id': policy_id, 'endpoint_id': ref['endpoint_id'], 'service_id': ref['service_id'], 'region_id': ref['region_id']})
    return matching_endpoints