from typing import List, Optional, Union
from ray._private.client_mode_hook import client_mode_hook
from ray._raylet import GcsClient
def internal_kv_get_gcs_client():
    return global_gcs_client