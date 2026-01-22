from typing import Dict, List, Optional, Union
import torch
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants
def set_devices(self, devices: List[DeviceType]):
    """
        Set local devices used by the TensorPipe RPC agent. When processing
        CUDA RPC requests, the TensorPipe RPC agent will properly synchronize
        CUDA streams for all devices in this ``List``.

        Args:
            devices (List of int, str, or torch.device): local devices used by
                the TensorPipe RPC agent.
        """
    self.devices = _to_device_list(devices)