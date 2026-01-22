import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
@retrying.retry(stop_max_attempt_number=5, retry_on_exception=_retry_on_bad_request, wait_exponential_multiplier=1000, wait_exponential_max=10000)
@handle_exceptions
def attachment_create(self, client, volume_id, connector=None, mountpoint=None, mode=None):
    """Create a volume attachment. This requires microversion >= 3.54.

        The attachment_create call was introduced in microversion 3.27. We
        need 3.54 as minimum here as we need attachment_complete to finish the
        attaching process and it which was introduced in version 3.44 and
        we also pass the attach mode which was introduced in version 3.54.

        :param client: cinderclient object
        :param volume_id: UUID of the volume on which to create the attachment.
        :param connector: host connector dict; if None, the attachment will
            be 'reserved' but not yet attached.
        :param mountpoint: Optional mount device name for the attachment,
            e.g. "/dev/vdb". This is only used if a connector is provided.
        :param mode: The mode in which the attachment is made i.e.
            read only(ro) or read/write(rw)
        :returns: a dict created from the
            cinderclient.v3.attachments.VolumeAttachment object with a backward
            compatible connection_info dict
        """
    if connector and mountpoint and ('mountpoint' not in connector):
        connector['mountpoint'] = mountpoint
    try:
        attachment_ref = client.attachments.create(volume_id, connector, mode=mode)
        return attachment_ref
    except cinder_exception.ClientException as ex:
        with excutils.save_and_reraise_exception():
            if getattr(ex, 'code', None) != 400:
                LOG.error(_LE('Create attachment failed for volume %(volume_id)s. Error: %(msg)s Code: %(code)s'), {'volume_id': volume_id, 'msg': str(ex), 'code': getattr(ex, 'code', None)})