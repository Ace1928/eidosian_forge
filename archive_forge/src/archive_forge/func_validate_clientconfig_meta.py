from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def validate_clientconfig_meta(clientconfig):
    """Validate the basics of the parsed clientconfig yaml for AIS Hub Feature Spec.

  Args:
    clientconfig: The data field of the YamlConfigFile.
  """
    if 'spec' not in clientconfig:
        raise exceptions.Error('Missing required field .spec')