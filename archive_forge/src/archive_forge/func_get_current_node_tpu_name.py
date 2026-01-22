import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_current_node_tpu_name() -> Optional[str]:
    """Return the name of the TPU pod that this worker node is a part of.

        For instance, if the TPU was created with name "my-tpu", this function
        will return "my-tpu".

        If created through the Ray cluster launcher, the
        name will typically be something like "ray-my-tpu-cluster-worker-aa946781-tpu".

        In case the TPU was created through KubeRay, we currently expect that the
        environment variable TPU_NAME is set per TPU pod slice, in which case
        this function will return the value of that environment variable.

        """
    try:
        tpu_name = os.getenv(GKE_TPU_NAME_ENV_VAR, None)
        if not tpu_name:
            tpu_name = _get_tpu_metadata(key=GCE_TPU_INSTANCE_ID_KEY)
        return tpu_name
    except ValueError as e:
        logging.debug('Could not get TPU name: %s', e)
        return None