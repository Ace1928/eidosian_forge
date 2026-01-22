import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
@handle_exceptions
def attachment_complete(self, client, attachment_id):
    """Marks a volume attachment complete.

        This call should be used to inform Cinder that a volume attachment is
        fully connected on the host so Cinder can apply the necessary state
        changes to the volume info in its database.

        :param client: cinderclient object
        :param attachment_id: UUID of the volume attachment to update.
        """
    try:
        client.attachments.complete(attachment_id)
    except cinder_exception.ClientException as ex:
        with excutils.save_and_reraise_exception():
            LOG.error(_LE('Complete attachment failed for attachment %(id)s. Error: %(msg)s Code: %(code)s'), {'id': attachment_id, 'msg': str(ex), 'code': getattr(ex, 'code', None)})