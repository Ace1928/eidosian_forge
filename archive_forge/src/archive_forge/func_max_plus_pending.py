from ray.dashboard.modules.metrics.dashboards.common import (
def max_plus_pending(max_resource, pending_resource):
    return f'({max_resource} or vector(0)) + ({pending_resource} or vector(0))'