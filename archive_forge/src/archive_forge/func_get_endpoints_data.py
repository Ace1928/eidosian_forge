import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def get_endpoints_data(self, service_type=None, interface=None, region_name=None, service_name=None, service_id=None, endpoint_id=None):
    """Fetch and filter endpoint data for the specified service(s).

        Returns endpoints for the specified service (or all) containing
        the specified type (or all) and region (or all) and service name.

        If there is no name in the service catalog the service_name check will
        be skipped.  This allows compatibility with services that existed
        before the name was available in the catalog.

        Valid interface types: `public` or `publicURL`,
                               `internal` or `internalURL`,
                               `admin` or 'adminURL`

        :param string service_type: Service type of the endpoint.
        :param interface: Type of endpoint. Can be a single value or a list
                          of values. If it's a list of values, they will be
                          looked for in order of preference.
        :param string region_name: Region of the endpoint.
        :param string service_name: The assigned name of the service.
        :param string service_id: The identifier of a service.
        :param string endpoint_id: The identifier of an endpoint.

        :returns: a list of matching EndpointData objects
        :rtype: list(`keystoneauth1.discover.EndpointData`)

        :returns: a dict, keyed by service_type, of lists of EndpointData
        """
    interfaces = self._get_interface_list(interface)
    matching_endpoints = {}
    for service in self.normalize_catalog():
        if service_type and (not discover._SERVICE_TYPES.is_match(service_type, service['type'])):
            continue
        if service_name and service['name'] and (service_name != service['name']):
            continue
        if service_id and service['id'] and (service_id != service['id']):
            continue
        matching_endpoints.setdefault(service['type'], [])
        for endpoint in service.get('endpoints', []):
            if interfaces and endpoint['interface'] not in interfaces:
                continue
            if region_name and region_name != endpoint['region_name']:
                continue
            if endpoint_id and endpoint_id != endpoint['id']:
                continue
            if not endpoint['url']:
                continue
            matching_endpoints[service['type']].append(discover.EndpointData(catalog_url=endpoint['url'], service_type=service['type'], service_name=service['name'], service_id=service['id'], interface=endpoint['interface'], region_name=endpoint['region_name'], endpoint_id=endpoint['id'], raw_endpoint=endpoint['raw_endpoint']))
    if not interfaces:
        return self._endpoints_by_type(service_type, matching_endpoints)
    ret = {}
    for matched_service_type, endpoints in matching_endpoints.items():
        if not endpoints:
            ret[matched_service_type] = []
            continue
        matches_by_interface = {}
        for endpoint in endpoints:
            matches_by_interface.setdefault(endpoint.interface, [])
            matches_by_interface[endpoint.interface].append(endpoint)
        best_interface = [i for i in interfaces if i in matches_by_interface.keys()][0]
        ret[matched_service_type] = matches_by_interface[best_interface]
    return self._endpoints_by_type(service_type, ret)