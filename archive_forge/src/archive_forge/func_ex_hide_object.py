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
def ex_hide_object(self, container_id, object_name):
    data = {}
    data['bucketId'] = container_id
    data['fileName'] = object_name
    resp = self.connection.request(action='b2_hide_file', data=data, method='POST')
    obj = self._to_object(item=resp.object, container=None)
    return obj