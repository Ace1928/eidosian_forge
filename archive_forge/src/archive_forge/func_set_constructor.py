from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def set_constructor(self, constructor):
    """Set object used for constructing new sequence instances.

    Args:
      constructor: Callable which can accept no arguments.  Must return
        an instance of the appropriate class for the container.
    """
    self.constructor = constructor