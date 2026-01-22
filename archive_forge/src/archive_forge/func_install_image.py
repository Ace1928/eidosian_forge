import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def install_image(self, path, ex_timeout=default_time_out, **ex_img_data):
    """
        Install a container image from a remote path. Not that the
        path currently is not used. Image data should be provided
        under the key 'ex_img_data'. Creating an image in LXD is an
        asynchronous operation

        :param path: Path to the container image
        :type  path: ``str``

        :param ex_timeout: Time to wait before signaling timeout
        :type  ex_timeout: ``int``

        :param ex_img_data: Dictionary describing the image data
        :type  ex_img_data: ``dict``

        :rtype: :class:`.ContainerImage`
        """
    if not ex_img_data:
        msg = 'Install an image for LXD requires specification of image_data'
        raise LXDAPIException(message=msg)
    data = ex_img_data['source']
    config = {'public': data.get('public', True), 'auto_update': data.get('auto_update', False), 'aliases': [data.get('aliases', {})], 'source': {'type': 'url', 'mode': 'pull', 'url': data['url']}}
    config = json.dumps(config)
    response = self.connection.request('/%s/images' % self.version, method='POST', data=config)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=100)
    try:
        id = response_dict['metadata']['id']
        req = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, ex_timeout)
        response = self.connection.request(req)
    except BaseHTTPError as err:
        lxd_exception = self._get_lxd_api_exception_for_error(err)
        if lxd_exception.message != 'not found':
            raise lxd_exception
    config = json.loads(config)
    if len(config['aliases']) != 0 and 'name' in config['aliases'][0]:
        image_alias = config['aliases'][0]['name']
    else:
        image_alias = config['source']['url'].split('/')[-1]
    has, fingerprint = self.ex_has_image(alias=image_alias)
    if not has:
        raise LXDAPIException(message='Image %s was not installed ' % image_alias)
    return self.ex_get_image(fingerprint=fingerprint)