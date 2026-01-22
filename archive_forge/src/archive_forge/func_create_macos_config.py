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
def create_macos_config(base_config, **kwargs):
    """Creates a MacOS ECP Config.

  Args:
    base_config: Optional parameter to use as a fallback for parameters that are
      not set in kwargs.
    **kwargs: MacOS config parameters. See go/enterprise-cert-config for valid
      variables.

  Returns:
    A dictionary object containing the ECP config.
  """
    if base_config:
        base_macos_config = base_config['cert_configs']['macos_keychain']
        base_libs_config = base_config['libs']
    else:
        base_macos_config = {}
        base_libs_config = {}
    ecp_config = KeyChainConfig(kwargs.get('issuer', None) or base_macos_config.get('issuer', None))
    lib_config = MacOSBinaryPathConfig(kwargs.get('ecp', None) or base_libs_config.get('ecp', None), kwargs.get('ecp_client', None) or base_libs_config.get('ecp_client', None), kwargs.get('tls_offload', None) or base_libs_config.get('tls_offload', None))
    return ({'macos_keychain': vars(ecp_config)}, {'libs': vars(lib_config)})