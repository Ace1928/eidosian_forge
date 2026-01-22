import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
@handle_exceptions
@retrying.retry(stop_max_attempt_number=5, retry_on_exception=_retry_on_internal_server_error)
def attachment_delete(self, client, attachment_id):
    try:
        client.attachments.delete(attachment_id)
    except cinder_exception.ClientException as ex:
        with excutils.save_and_reraise_exception():
            LOG.error(_LE('Delete attachment failed for attachment %(id)s. Error: %(msg)s Code: %(code)s'), {'id': attachment_id, 'msg': str(ex), 'code': getattr(ex, 'code', None)})