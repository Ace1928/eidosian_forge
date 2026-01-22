import logging
import socket
import tempfile
from oslo_config import cfg
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _
def validate_buffering(buffer_dir):
    if buffer_dir is None:
        msg = _('Configuration option "swift_upload_buffer_dir" is not set. Please set it to a valid path to buffer during Swift uploads.')
        raise exceptions.BadStoreConfiguration(store_name='swift', reason=msg)
    try:
        _tmpfile = tempfile.TemporaryFile(dir=buffer_dir)
    except OSError as err:
        msg = _('Unable to use buffer directory set with "swift_upload_buffer_dir". Error: %s') % encodeutils.exception_to_unicode(err)
        raise exceptions.BadStoreConfiguration(store_name='swift', reason=msg)
    else:
        _tmpfile.close()
        return True