from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_containerinstance(self):
    """
        Creates or updates a container service with the specified configuration of orchestrator, masters, and agents.

        :return: deserialized container instance state dictionary
        """
    self.log('Creating / Updating the container instance {0}'.format(self.name))
    registry_credentials = None
    if self.registry_login_server is not None:
        registry_credentials = [self.cgmodels.ImageRegistryCredential(server=self.registry_login_server, username=self.registry_username, password=self.registry_password)]
    ip_address = None
    containers = []
    all_ports = set([])
    for container_def in self.containers:
        name = container_def.get('name')
        image = container_def.get('image')
        memory = container_def.get('memory')
        cpu = container_def.get('cpu')
        commands = container_def.get('commands')
        ports = []
        variables = []
        volume_mounts = []
        port_list = container_def.get('ports')
        if port_list:
            for port in port_list:
                all_ports.add(port)
                ports.append(self.cgmodels.ContainerPort(port=port))
        variable_list = container_def.get('environment_variables')
        if variable_list:
            for variable in variable_list:
                variables.append(self.cgmodels.EnvironmentVariable(name=variable.get('name'), value=variable.get('value') if not variable.get('is_secure') else None, secure_value=variable.get('value') if variable.get('is_secure') else None))
        volume_mounts_list = container_def.get('volume_mounts')
        if volume_mounts_list:
            for volume_mount in volume_mounts_list:
                volume_mounts.append(self.cgmodels.VolumeMount(name=volume_mount.get('name'), mount_path=volume_mount.get('mount_path'), read_only=volume_mount.get('read_only')))
        containers.append(self.cgmodels.Container(name=name, image=image, resources=self.cgmodels.ResourceRequirements(requests=self.cgmodels.ResourceRequests(memory_in_gb=memory, cpu=cpu)), ports=ports, command=commands, environment_variables=variables, volume_mounts=volume_mounts))
    if self.ip_address is not None:
        if len(all_ports) > 0:
            ports = []
            for port in all_ports:
                ports.append(self.cgmodels.Port(port=port, protocol='TCP'))
            ip_address = self.cgmodels.IpAddress(ports=ports, dns_name_label=self.dns_name_label, type=self.ip_address)
    subnet_ids = None
    if self.subnet_ids is not None:
        subnet_ids = [self.cgmodels.ContainerGroupSubnetId(id=item) for item in self.subnet_ids]
    parameters = self.cgmodels.ContainerGroup(location=self.location, containers=containers, image_registry_credentials=registry_credentials, restart_policy=_snake_to_camel(self.restart_policy, True) if self.restart_policy else None, ip_address=ip_address, os_type=self.os_type, subnet_ids=subnet_ids, volumes=self.volumes, tags=self.tags)
    try:
        response = self.containerinstance_client.container_groups.begin_create_or_update(resource_group_name=self.resource_group, container_group_name=self.name, container_group=parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error when creating ACI {0}: {1}'.format(self.name, exc.message or str(exc)))
    return response.as_dict()