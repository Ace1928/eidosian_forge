import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_image(self, dry_run: bool=False, image: NodeImage=None, perm_to_launch_addition_account_ids: List[str]=None, perm_to_launch_addition_global_permission: bool=None, perm_to_launch_removals_account_ids: List[str]=None, perm_to_launch_removals_global_permission: bool=None):
    """
        Modifies the specified attribute of an Outscale machine image (OMI).
        You can specify only one attribute at a time. You can modify
        the permissions to access the OMI by adding or removing account
        IDs or groups. You can share an OMI with a user that is in the
        same Region. The user can create a copy of the OMI you shared,
        obtaining all the rights for the copy of the OMI.
        For more information, see CreateImage.

        :param      image: The ID of the OMI to export. (required)
        :type       image: ``NodeImage``

        :param      perm_to_launch_addition_account_ids: The account
        ID of one or more users who have permissions for the resource.
        :type       perm_to_launch_addition_account_ids:
        ``list`` of ``dict``

        :param      perm_to_launch_addition_global_permission:
        If true, the resource is public. If false, the resource is private.
        :type       perm_to_launch_addition_global_permission:
        ``boolean``

        :param      perm_to_launch_removals_account_ids: The account
        ID of one or more users who have permissions for the resource.
        :type       perm_to_launch_removals_account_ids: ``list`` of ``dict``

        :param      perm_to_launch_removals_global_permission: If true,
        the resource is public. If false, the resource is private.
        :type       perm_to_launch_removals_global_permission: ``boolean``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: the new image
        :rtype: ``dict``
        """
    action = 'UpdateImage'
    data = {'DryRun': dry_run, 'PermissionsToLaunch': {'Additions': {}, 'Removals': {}}}
    if image is not None:
        data.update({'ImageId': image.id})
    if perm_to_launch_addition_account_ids is not None:
        data['PermissionsToLaunch']['Additions'].update({'AccountIds': perm_to_launch_addition_account_ids})
    if perm_to_launch_addition_global_permission is not None:
        data['PermissionsToLaunch']['Additions'].update({'GlobalPermission': perm_to_launch_addition_global_permission})
    if perm_to_launch_removals_account_ids is not None:
        data['PermissionsToLaunch']['Removals'].update({'AccountIds': perm_to_launch_removals_account_ids})
    if perm_to_launch_removals_global_permission is not None:
        data['PermissionsToLaunch']['Removals'].update({'GlobalPermission': perm_to_launch_removals_global_permission})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Image']
    return response.json()