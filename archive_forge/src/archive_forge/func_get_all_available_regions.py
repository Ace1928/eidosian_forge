import boto.vendored.regions.regions as _regions
def get_all_available_regions(self, service_name):
    """Get all the regions a service is available in.

        :type service_name: str
        :param service_name: The service to look up.

        :rtype: list of str
        :return: A list of all the regions the given service is available in.
        """
    return self._resolver.get_all_available_regions(service_name)