import urllib.parse as urlparse
from oslo_config import cfg
from glance.i18n import _

    Order image location list.

    :param locations: The original image location list.
    :param uri_key: The key name for location URI in image location dictionary.
    :returns: The image location list with preferred store type order.
    