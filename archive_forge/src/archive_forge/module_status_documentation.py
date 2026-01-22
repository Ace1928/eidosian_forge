from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun import component_status
import six
Instantiate a new ModuleStatus from JSON.

    Args:
      name: the name of the Module
      json_map: a JSON dict mapping component name to the JSON representation of
        ComponentStatus (see ComponentStatus.FromJSON)

    Returns:
      a ModuleStatus object
    