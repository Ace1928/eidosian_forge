import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_accelerator_ids(self) -> Dict[str, List[str]]:
    """
        Get the current worker's visible accelerator ids.

        Returns:
            A dictionary keyed by the accelerator resource name. The values are a list
            of ids `{'GPU': ['0', '1'], 'neuron_cores': ['0', '1'],
            'TPU': ['0', '1']}`.
        """
    worker = self.worker
    worker.check_connected()
    ids_dict: Dict[str, List[str]] = {}
    for accelerator_resource_name in ray._private.accelerators.get_all_accelerator_resource_names():
        accelerator_ids = worker.get_accelerator_ids_for_accelerator_resource(accelerator_resource_name, f'^{accelerator_resource_name}_group_[0-9A-Za-z]+$')
        ids_dict[accelerator_resource_name] = accelerator_ids
    return ids_dict