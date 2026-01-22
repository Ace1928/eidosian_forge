from oslo_config import cfg
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_log import log as logging
from oslo_utils import units
from glance.common import exception
from glance.db.sqlalchemy import api as db
from glance.i18n import _LE
def enforce_image_staging_total(context, project_id, delta=0):
    """Enforce the image_stage_total quota.

    This enforces the total size of all images stored in staging areas
    for the supplied project_id.
    """
    _enforce_one(context, project_id, QUOTA_IMAGE_STAGING_TOTAL, lambda: db.user_get_staging_usage(context, project_id) // units.Mi, delta=delta)