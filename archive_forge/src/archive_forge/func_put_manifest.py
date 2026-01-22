from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import concurrent.futures
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list as image_list
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def put_manifest(self, image, use_digest=False):
    """Upload the manifest for this image."""
    if use_digest:
        tag_or_digest = image.digest()
    else:
        tag_or_digest = _tag_or_digest(self._name)
    self._transport.Request('{base_url}/manifests/{tag_or_digest}'.format(base_url=self._base_url(), tag_or_digest=tag_or_digest), method='PUT', body=image.manifest(), content_type=image.media_type(), accepted_codes=[six.moves.http_client.OK, six.moves.http_client.CREATED, six.moves.http_client.ACCEPTED])