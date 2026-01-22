from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def get_configurations(spec):
    """Extracts a dictionary of deployment configuration by component name.

  Args:
    spec: A hub membership spec.

  Returns:
    A dictionary mapping component name to configuration object.
  """
    return {cfg.key: cfg.value for cfg in spec.policycontroller.policyControllerHubConfig.deploymentConfigs.additionalProperties}