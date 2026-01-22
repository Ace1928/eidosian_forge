import boto.vendored.regions.regions as _regions
def build_static_endpoints(self, service_names=None):
    """Build a set of static endpoints in the legacy boto2 format.

        :param service_names: The names of the services to build. They must
            use the names that boto2 uses, not boto3, e.g "ec2containerservice"
            and not "ecs". If no service names are provided, all available
            services will be built.

        :return: A dict consisting of::
            {"service": {"region": "full.host.name"}}
        """
    if service_names is None:
        service_names = self._resolver.get_available_services()
    static_endpoints = {}
    for name in service_names:
        endpoints_for_service = self._build_endpoints_for_service(name)
        if endpoints_for_service:
            static_endpoints[name] = endpoints_for_service
    self._handle_special_cases(static_endpoints)
    return static_endpoints