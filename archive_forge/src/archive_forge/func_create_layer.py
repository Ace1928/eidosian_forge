import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def create_layer(self, stack_id, type, name, shortname, attributes=None, custom_instance_profile_arn=None, custom_security_group_ids=None, packages=None, volume_configurations=None, enable_auto_healing=None, auto_assign_elastic_ips=None, auto_assign_public_ips=None, custom_recipes=None, install_updates_on_boot=None, use_ebs_optimized_instances=None, lifecycle_event_configuration=None):
    """
        Creates a layer. For more information, see `How to Create a
        Layer`_.


        You should use **CreateLayer** for noncustom layer types such
        as PHP App Server only if the stack does not have an existing
        layer of that type. A stack can have at most one instance of
        each noncustom layer; if you attempt to create a second
        instance, **CreateLayer** fails. A stack can have an arbitrary
        number of custom layers, so you can call **CreateLayer** as
        many times as you like for that layer type.


        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type stack_id: string
        :param stack_id: The layer stack ID.

        :type type: string
        :param type: The layer type. A stack cannot have more than one built-in
            layer of the same type. It can have any number of custom layers.

        :type name: string
        :param name: The layer name, which is used by the console.

        :type shortname: string
        :param shortname: The layer short name, which is used internally by AWS
            OpsWorks and by Chef recipes. The short name is also used as the
            name for the directory where your app files are installed. It can
            have a maximum of 200 characters, which are limited to the
            alphanumeric characters, '-', '_', and '.'.

        :type attributes: map
        :param attributes: One or more user-defined key/value pairs to be added
            to the stack attributes.

        :type custom_instance_profile_arn: string
        :param custom_instance_profile_arn: The ARN of an IAM profile that to
            be used for the layer's EC2 instances. For more information about
            IAM ARNs, see `Using Identifiers`_.

        :type custom_security_group_ids: list
        :param custom_security_group_ids: An array containing the layer custom
            security group IDs.

        :type packages: list
        :param packages: An array of `Package` objects that describe the layer
            packages.

        :type volume_configurations: list
        :param volume_configurations: A `VolumeConfigurations` object that
            describes the layer's Amazon EBS volumes.

        :type enable_auto_healing: boolean
        :param enable_auto_healing: Whether to disable auto healing for the
            layer.

        :type auto_assign_elastic_ips: boolean
        :param auto_assign_elastic_ips: Whether to automatically assign an
            `Elastic IP address`_ to the layer's instances. For more
            information, see `How to Edit a Layer`_.

        :type auto_assign_public_ips: boolean
        :param auto_assign_public_ips: For stacks that are running in a VPC,
            whether to automatically assign a public IP address to the layer's
            instances. For more information, see `How to Edit a Layer`_.

        :type custom_recipes: dict
        :param custom_recipes: A `LayerCustomRecipes` object that specifies the
            layer custom recipes.

        :type install_updates_on_boot: boolean
        :param install_updates_on_boot:
        Whether to install operating system and package updates when the
            instance boots. The default value is `True`. To control when
            updates are installed, set this value to `False`. You must then
            update your instances manually by using CreateDeployment to run the
            `update_dependencies` stack command or manually running `yum`
            (Amazon Linux) or `apt-get` (Ubuntu) on the instances.


        We strongly recommend using the default value of `True`, to ensure that
            your instances have the latest security updates.

        :type use_ebs_optimized_instances: boolean
        :param use_ebs_optimized_instances: Whether to use Amazon EBS-optimized
            instances.

        :type lifecycle_event_configuration: dict
        :param lifecycle_event_configuration: A LifeCycleEventConfiguration
            object that you can use to configure the Shutdown event to specify
            an execution timeout and enable or disable Elastic Load Balancer
            connection draining.

        """
    params = {'StackId': stack_id, 'Type': type, 'Name': name, 'Shortname': shortname}
    if attributes is not None:
        params['Attributes'] = attributes
    if custom_instance_profile_arn is not None:
        params['CustomInstanceProfileArn'] = custom_instance_profile_arn
    if custom_security_group_ids is not None:
        params['CustomSecurityGroupIds'] = custom_security_group_ids
    if packages is not None:
        params['Packages'] = packages
    if volume_configurations is not None:
        params['VolumeConfigurations'] = volume_configurations
    if enable_auto_healing is not None:
        params['EnableAutoHealing'] = enable_auto_healing
    if auto_assign_elastic_ips is not None:
        params['AutoAssignElasticIps'] = auto_assign_elastic_ips
    if auto_assign_public_ips is not None:
        params['AutoAssignPublicIps'] = auto_assign_public_ips
    if custom_recipes is not None:
        params['CustomRecipes'] = custom_recipes
    if install_updates_on_boot is not None:
        params['InstallUpdatesOnBoot'] = install_updates_on_boot
    if use_ebs_optimized_instances is not None:
        params['UseEbsOptimizedInstances'] = use_ebs_optimized_instances
    if lifecycle_event_configuration is not None:
        params['LifecycleEventConfiguration'] = lifecycle_event_configuration
    return self.make_request(action='CreateLayer', body=json.dumps(params))