from typing import Set, Optional
from ray._private.accelerators.accelerator import AcceleratorManager
from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager
from ray._private.accelerators.intel_gpu import IntelGPUAcceleratorManager
from ray._private.accelerators.tpu import TPUAcceleratorManager
from ray._private.accelerators.neuron import NeuronAcceleratorManager
from ray._private.accelerators.hpu import HPUAcceleratorManager
from ray._private.accelerators.npu import NPUAcceleratorManager
def get_all_accelerator_resource_names() -> Set[str]:
    """Get all resource names for accelerators."""
    return {accelerator_manager.get_resource_name() for accelerator_manager in get_all_accelerator_managers()}