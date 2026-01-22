from __future__ import (absolute_import, division, print_function)
def get_routes_for_namespace(self, client, name, namespace):
    self.check_kubernetes_collection()
    v1_route = client.resources.get(api_version='route.openshift.io/v1', kind='Route')
    try:
        obj = v1_route.get(namespace=namespace)
    except DynamicApiError as exc:
        self.display.debug(exc)
        raise K8sInventoryException('Error fetching Routes list: %s' % format_dynamic_api_exc(exc))
    namespace_group = 'namespace_{0}'.format(namespace)
    namespace_routes_group = '{0}_routes'.format(namespace_group)
    self.inventory.add_group(name)
    self.inventory.add_group(namespace_group)
    self.inventory.add_child(name, namespace_group)
    self.inventory.add_group(namespace_routes_group)
    self.inventory.add_child(namespace_group, namespace_routes_group)
    for route in obj.items:
        route_name = route.metadata.name
        route_annotations = {} if not route.metadata.annotations else dict(route.metadata.annotations)
        self.inventory.add_host(route_name)
        if route.metadata.labels:
            for key, value in route.metadata.labels:
                group_name = 'label_{0}_{1}'.format(key, value)
                self.inventory.add_group(group_name)
                self.inventory.add_child(group_name, route_name)
            route_labels = dict(route.metadata.labels)
        else:
            route_labels = {}
        self.inventory.add_child(namespace_routes_group, route_name)
        self.inventory.set_variable(route_name, 'labels', route_labels)
        self.inventory.set_variable(route_name, 'annotations', route_annotations)
        self.inventory.set_variable(route_name, 'cluster_name', route.metadata.clusterName)
        self.inventory.set_variable(route_name, 'object_type', 'route')
        self.inventory.set_variable(route_name, 'self_link', route.metadata.selfLink)
        self.inventory.set_variable(route_name, 'resource_version', route.metadata.resourceVersion)
        self.inventory.set_variable(route_name, 'uid', route.metadata.uid)
        if route.spec.host:
            self.inventory.set_variable(route_name, 'host', route.spec.host)
        if route.spec.path:
            self.inventory.set_variable(route_name, 'path', route.spec.path)
        if hasattr(route.spec.port, 'targetPort') and route.spec.port.targetPort:
            self.inventory.set_variable(route_name, 'port', dict(route.spec.port))