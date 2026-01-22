import abc
import copy
from ansible.module_utils.six import raise_from
import importlib
import os
from ansible.module_utils.basic import AnsibleModule
def openstack_cloud_from_module(self):
    """Sets up connection to cloud using provided options. Checks if all
           provided variables are supported for the used SDK version.
        """
    try:
        sdk = importlib.import_module('openstack')
        self.sdk_version = sdk.version.__version__
    except ImportError:
        self.fail_json(msg='openstacksdk is required for this module')
    try:
        ensure_compatibility(self.sdk_version, self.module_min_sdk_version, self.module_max_sdk_version)
    except ImportError as e:
        self.fail_json(msg='Incompatible openstacksdk library found: {error}.'.format(error=str(e)))
    for param in self.argument_spec:
        if self.params[param] is not None and 'min_ver' in self.argument_spec[param] and (StrictVersion(self.sdk_version) < self.argument_spec[param]['min_ver']):
            self.fail_json(msg="To use parameter '{param}' with module '{module}', the installed version of the openstacksdk library MUST be >={min_version}.".format(min_version=self.argument_spec[param]['min_ver'], param=param, module=self.module_name))
        if self.params[param] is not None and 'max_ver' in self.argument_spec[param] and (StrictVersion(self.sdk_version) > self.argument_spec[param]['max_ver']):
            self.fail_json(msg="To use parameter '{param}' with module '{module}', the installed version of the openstacksdk library MUST be <={max_version}.".format(max_version=self.argument_spec[param]['max_ver'], param=param, module=self.module_name))
    cloud_config = self.params.pop('cloud', None)
    if isinstance(cloud_config, dict):
        fail_message = 'A cloud config dict was provided to the cloud parameter but also a value was provided for {param}. If a cloud config dict is provided, {param} should be excluded.'
        for param in ('auth', 'region_name', 'validate_certs', 'ca_cert', 'client_key', 'api_timeout', 'auth_type'):
            if self.params[param] is not None:
                self.fail_json(msg=fail_message.format(param=param))
        if self.params['interface'] != 'public':
            self.fail_json(msg=fail_message.format(param='interface'))
    else:
        cloud_config = dict(cloud=cloud_config, auth_type=self.params['auth_type'], auth=self.params['auth'], region_name=self.params['region_name'], verify=self.params['validate_certs'], cacert=self.params['ca_cert'], key=self.params['client_key'], api_timeout=self.params['api_timeout'], interface=self.params['interface'])
    try:
        return (sdk, sdk.connect(**cloud_config))
    except sdk.exceptions.SDKException as e:
        self.fail_json(msg=str(e))