from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def create_ecp_config(config_type, base_config=None, **kwargs):
    """Creates an ECP Config.

  Args:
    config_type: An ConfigType Enum that describes the type of ECP config.
    base_config: Optional parameter to use as a fallback for parameters that are
      not set in kwargs.
    **kwargs: config parameters. See go/enterprise-cert-config for valid
      variables.

  Returns:
    A dictionary object containing the ECP config.
  Raises:
    ECPConfigError: No valid config_type is specified.
  """
    if config_type == ConfigType.PKCS11:
        ecp_config, libs_config = create_linux_config(base_config, **kwargs)
    elif config_type == ConfigType.KEYCHAIN:
        ecp_config, libs_config = create_macos_config(base_config, **kwargs)
    elif config_type == ConfigType.MYSTORE:
        ecp_config, libs_config = create_windows_config(base_config, **kwargs)
    else:
        raise ECPConfigError('Unknown config_type {} passed to create enterprise certificate configuration. Valid options are: [PKCS11, KEYCHAIN, MYSTORE]'.format(config_type))
    return {'cert_configs': ecp_config, **libs_config}