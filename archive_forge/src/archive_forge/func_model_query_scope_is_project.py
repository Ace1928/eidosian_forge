import functools
from oslo_db import exception as db_exc
from oslo_utils import excutils
import sqlalchemy
from sqlalchemy.ext import associationproxy
from sqlalchemy.orm import exc
from sqlalchemy.orm import properties
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import exceptions as n_exc
def model_query_scope_is_project(context, model):
    """Determine if a model should be scoped to a project.

    :param context: The context to check for admin and advsvc rights.
    :param model: The model to check the project_id of.
    :returns: True if the context is not admin and not advsvc and the model
        has a project_id. False otherwise.
    """
    if not hasattr(model, 'project_id'):
        return False
    if context.is_service_role:
        return False
    return not context.is_admin