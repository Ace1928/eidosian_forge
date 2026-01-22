import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_snapshot_export_task(self, osu_export_disk_image_format: str=None, osu_export_api_key_id: str=None, osu_export_api_secret_key: str=None, osu_export_bucket: str=None, osu_export_manifest_url: str=None, osu_export_prefix: str=None, snapshot: VolumeSnapshot=None, dry_run: bool=False):
    """
        Exports a snapshot to an Object Storage Unit (OSU) bucket.
        This action enables you to create a backup of your snapshot or to copy
        it to another account. You, or other users you send a pre-signed URL
        to, can then download this snapshot from the OSU bucket using
        the CreateSnapshot method.
        This procedure enables you to copy a snapshot between accounts within
        the same Region or in different Regions. To copy a snapshot within
        the same Region, you can also use the CreateSnapshot direct method.
        The copy of the source snapshot is independent and belongs to you.

        :param      osu_export_disk_image_format: The format of the export
        disk (qcow2 | vdi | vmdk). (required)
        :type       osu_export_disk_image_format: ``str``

        :param      osu_export_api_key_id: The API key of the OSU account
        that enables you to access the bucket.
        :type       osu_export_api_key_id   : ``str``

        :param      osu_export_api_secret_key: The secret key of the OSU
        account that enables you to access the bucket.
        :type       osu_export_api_secret_key   : ``str``

        :param      osu_export_bucket: The name of the OSU bucket you want
        to export the object to.  (required)
        :type       osu_export_bucket   : ``str``

        :param      osu_export_manifest_url: The URL of the manifest file.
        :type       osu_export_manifest_url   : ``str``

        :param      osu_export_prefix: The prefix for the key of the OSU
        object. This key follows this format: prefix +
        object_export_task_id + '.' + disk_image_format.
        :type       osu_export_prefix   : ``str``

        :param      snapshot: The ID of the snapshot to export. (required)
        :type       snapshot   : ``VolumeSnapshot``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run   : ``bool``

        :return: the created snapshot export task
        :rtype: ``dict``
        """
    action = 'CreateSnapshotExportTask'
    data = {'DryRun': dry_run, 'OsuExport': {'OsuApiKey': {}}}
    if snapshot is not None:
        data.update({'SnapshotId': snapshot.id})
    if osu_export_disk_image_format is not None:
        data['OsuExport'].update({'DiskImageFormat': osu_export_disk_image_format})
    if osu_export_bucket is not None:
        data['OsuExport'].update({'OsuBucket': osu_export_bucket})
    if osu_export_manifest_url is not None:
        data['OsuExport'].update({'OsuManifestUrl': osu_export_manifest_url})
    if osu_export_prefix is not None:
        data['OsuExport'].update({'OsuPrefix': osu_export_prefix})
    if osu_export_api_key_id is not None:
        data['OsuExport']['OsuApiKey'].update({'ApiKeyId': osu_export_api_key_id})
    if osu_export_api_secret_key is not None:
        data['OsuExport']['OsuApiKey'].update({'SecretKey': osu_export_api_secret_key})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['SnapshotExportTask']
    return response.json()