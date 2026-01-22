from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread

      Provides a Python interface to GPU management and monitoring functions.

      This is a wrapper around the NVML library.
      For information about the NVML library, see the NVML developer page
      http://developer.nvidia.com/nvidia-management-library-nvml

      Examples:
      ---------------------------------------------------------------------------
      For all elements as a list of dictionaries.  Similiar to nvisia-smi -q -x

      $ DeviceQuery()

      ---------------------------------------------------------------------------
      For a list of filtered dictionary elements by string name.
      Similiar ot nvidia-smi --query-gpu=pci.bus_id,memory.total,memory.free
      See help_query_gpu.txt or DeviceQuery("--help_query_gpu") for available filter elements

      $ DeviceQuery("pci.bus_id,memory.total,memory.free")

      ---------------------------------------------------------------------------
      For a list of filtered dictionary elements by enumeration value.
      See help_query_gpu.txt or DeviceQuery("--help-query-gpu") for available filter elements

      $ DeviceQuery([NVSMI_PCI_BUS_ID, NVSMI_MEMORY_TOTAL, NVSMI_MEMORY_FREE])

      