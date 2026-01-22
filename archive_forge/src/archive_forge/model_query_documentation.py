from oslo_db.sqlalchemy import utils as sa_utils
from sqlalchemy.orm import lazyload
from sqlalchemy import sql, or_, and_
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import constants
from neutron_lib.db import utils as db_utils
from neutron_lib import exceptions as n_exc
from neutron_lib.objects import utils as obj_utils
from neutron_lib.utils import helpers
Get the count for a specific collection.

    :param context: The context to use for the DB session.
    :param model: The model for the query.
    :param filters: The filters to apply.
    :param query_field: Column, in string format, from the "model"; the query
                        will return only this parameter instead of the full
                        model columns.
    :returns: The number of objects for said model with filters applied.
    