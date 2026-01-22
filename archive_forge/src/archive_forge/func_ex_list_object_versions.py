import base64
import hashlib
from libcloud.utils.py3 import b, next, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.utils.escape import sanitize_object_name
from libcloud.storage.types import ObjectDoesNotExistError, ContainerDoesNotExistError
from libcloud.storage.providers import Provider
def ex_list_object_versions(self, container_id, ex_start_file_name=None, ex_start_file_id=None, ex_max_file_count=None):
    params = {}
    params['bucketId'] = container_id
    if ex_start_file_name:
        params['startFileName'] = ex_start_file_name
    if ex_start_file_id:
        params['startFileId'] = ex_start_file_id
    if ex_max_file_count:
        params['maxFileCount'] = ex_max_file_count
    resp = self.connection.request(action='b2_list_file_versions', params=params, method='GET')
    objects = self._to_objects(data=resp.object, container=None)
    return objects