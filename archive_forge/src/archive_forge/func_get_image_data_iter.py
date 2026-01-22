import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
def get_image_data_iter(uri):
    """Returns iterable object either for local file or uri

    :param uri: uri (remote or local) to the datasource we want to iterate

    Validation/sanitization of the uri is expected to happen before we get
    here.
    """
    if uri.startswith('file://'):
        uri = uri.split('file://')[-1]
        return open(uri, 'rb')
    return urllib.request.urlopen(uri)