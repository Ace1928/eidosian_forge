import warnings
from typing import Dict, Optional
from ray.air.execution.resources.request import ResourceRequest
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.placement_group import placement_group
@DeveloperAPI
def resource_dict_to_pg_factory(spec: Optional[Dict[str, float]]=None):
    """Translates resource dict into PlacementGroupFactory."""
    spec = spec or {'cpu': 1}
    spec = spec.copy()
    cpus = spec.pop('cpu', spec.pop('CPU', 0.0))
    gpus = spec.pop('gpu', spec.pop('GPU', 0.0))
    memory = spec.pop('memory', 0.0)
    bundle = {k: v for k, v in spec.pop('custom_resources', {}).items()}
    if not bundle:
        bundle = spec
    bundle.update({'CPU': cpus, 'GPU': gpus, 'memory': memory})
    return PlacementGroupFactory([bundle])