import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_node_types(self, bsu_optimized: bool=None, memory_sizes: List[int]=None, vcore_counts: List[int]=None, vm_type_names: List[str]=None, volume_counts: List[int]=None, volume_sizes: List[int]=None, dry_run: bool=False):
    """
        Lists one or more predefined VM types.

        :param      bsu_optimized: Indicates whether the VM is optimized for
        BSU I/O.
        :type       bsu_optimized: ``bool``

        :param      memory_sizes: The amounts of memory, in gibibytes (GiB).
        :type       memory_sizes: ``list`` of ``int``

        :param      vcore_counts: The numbers of vCores.
        :type       vcore_counts: ``list`` of ``int``

        :param      vm_type_names: The names of the VM types. For more
        information, see Instance Types.
        :type       vm_type_names: ``list`` of ``str``

        :param      volume_counts: The maximum number of ephemeral storage
        disks.
        :type       volume_counts: ``list`` of ``int``


        :param      volume_sizes: The size of one ephemeral storage disk,
        in gibibytes (GiB).
        :type       volume_sizes: ``list`` of ``int``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: list of vm types
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadVmTypes'
    data = {'Filters': {}, 'DryRun': dry_run}
    if bsu_optimized is not None:
        data['Filters'].update({'BsuOptimized': bsu_optimized})
    if memory_sizes is not None:
        data['Filters'].update({'MemorySizes': memory_sizes})
    if vcore_counts is not None:
        data['Filters'].update({'VcoreCounts': vcore_counts})
    if vm_type_names is not None:
        data['Filters'].update({'VmTypeNames': vm_type_names})
    if volume_counts is not None:
        data['Filters'].update({'VolumeCounts': volume_counts})
    if volume_sizes is not None:
        data['Filters'].update({'VolumeSizes': volume_sizes})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['VmTypes']
    return response.json()