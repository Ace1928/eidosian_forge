from keystoneauth1 import adapter
from keystoneauth1.loading import _utils
from keystoneauth1.loading import base
def process_conf_options(confgrp, kwargs):
    """Set Adapter constructor kwargs based on conf options.

    :param oslo_config.cfg.GroupAttr confgrp: Config object group containing
            options to inspect.
    :param dict kwargs: Keyword arguments suitable for the constructor of
            keystoneauth1.adapter.Adapter. Will be modified by this method.
            Values already set remain unaffected.
    :raise TypeError: If invalid conf option values or combinations are found.
    """
    if confgrp.valid_interfaces and getattr(confgrp, 'interface', None):
        raise TypeError('interface and valid_interfaces are mutually exclusive. Please use valid_interfaces.')
    if confgrp.valid_interfaces:
        for iface in confgrp.valid_interfaces:
            if iface not in ('public', 'internal', 'admin'):
                raise TypeError("'{iface}' is not a valid value for valid_interfaces. Valid valies are public, internal or admin".format(iface=iface))
        kwargs.setdefault('interface', confgrp.valid_interfaces)
    elif hasattr(confgrp, 'interface'):
        kwargs.setdefault('interface', confgrp.interface)
    kwargs.setdefault('service_type', confgrp.service_type)
    kwargs.setdefault('service_name', confgrp.service_name)
    kwargs.setdefault('region_name', confgrp.region_name)
    kwargs.setdefault('endpoint_override', confgrp.endpoint_override)
    kwargs.setdefault('version', confgrp.version)
    kwargs.setdefault('min_version', confgrp.min_version)
    kwargs.setdefault('max_version', confgrp.max_version)
    if kwargs['version'] and (kwargs['max_version'] or kwargs['min_version']):
        raise TypeError('version is mutually exclusive with min_version and max_version')
    kwargs.setdefault('connect_retries', confgrp.connect_retries)
    kwargs.setdefault('connect_retry_delay', confgrp.connect_retry_delay)
    kwargs.setdefault('status_code_retries', confgrp.status_code_retries)
    kwargs.setdefault('status_code_retry_delay', confgrp.status_code_retry_delay)
    kwargs.setdefault('retriable_status_codes', confgrp.retriable_status_codes)