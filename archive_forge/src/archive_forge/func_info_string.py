import logging
from ray._private.ray_constants import DEBUG_AUTOSCALING_STATUS_LEGACY
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
def info_string(autoscaler, nodes):
    suffix = ''
    if autoscaler.updaters:
        suffix += ' ({} updating)'.format(len(autoscaler.updaters))
    if autoscaler.num_failed_updates:
        suffix += ' ({} failed to update)'.format(len(autoscaler.num_failed_updates))
    return '{} nodes{}'.format(len(nodes), suffix)