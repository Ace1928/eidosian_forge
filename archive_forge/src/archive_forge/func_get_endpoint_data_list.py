import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def get_endpoint_data_list(self, service_type=None, interface='public', region_name=None, service_name=None, service_id=None, endpoint_id=None):
    """Fetch a flat list of matching EndpointData objects.

        Fetch the endpoints from the service catalog for a particular
        endpoint attribute. If no attribute is given, return the first
        endpoint of the specified type.

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
        """
    endpoints = self.get_endpoints_data(service_type=service_type, interface=interface, region_name=region_name, service_name=service_name, service_id=service_id, endpoint_id=endpoint_id)
    return [endpoint for data in endpoints.values() for endpoint in data]