import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
def format_image_notification(image):
    """
    Given a glance.domain.Image object, return a dictionary of relevant
    notification information. We purposely do not include 'location'
    as it may contain credentials.
    """
    return {'id': image.image_id, 'name': image.name, 'status': image.status, 'created_at': timeutils.isotime(image.created_at), 'updated_at': timeutils.isotime(image.updated_at), 'min_disk': image.min_disk, 'min_ram': image.min_ram, 'protected': image.protected, 'checksum': image.checksum, 'owner': image.owner, 'disk_format': image.disk_format, 'container_format': image.container_format, 'size': image.size, 'virtual_size': image.virtual_size, 'is_public': image.visibility == 'public', 'visibility': image.visibility, 'properties': dict(image.extra_properties), 'tags': list(image.tags), 'deleted': False, 'deleted_at': None}