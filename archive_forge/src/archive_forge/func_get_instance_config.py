from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import path_simplifier
import six
def get_instance_config(self, igm_ref, instance_ref):
    """Returns instance config for given reference (uses simple cache)."""
    per_instance_config_key = self._build_key(igm_ref=igm_ref, instance_ref=instance_ref)
    if self._key_of_cached_per_instance_config != per_instance_config_key:
        self._cached_per_instance_config = self._do_get_instance_config(igm_ref=igm_ref, instance_ref=instance_ref)
        self._key_of_cached_per_instance_config = per_instance_config_key
    return self._cached_per_instance_config