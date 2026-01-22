import os
from cursive import exception as cursive_exception
import glance_store
from glance_store import backend
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob.exc
import glance.api.policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import trust_auth
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI
import glance.notifier
from glance.quota import keystone as ks_quota

        Restore the image to queued status and remove data from staging.

        :param image_repo: The instance of ImageRepo
        :param image: The image will be restored
        :param staging_store: The store used for staging
        