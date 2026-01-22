from oslo_utils import timeutils
import sqlalchemy as sa
from sqlalchemy import event  # noqa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import attributes
from sqlalchemy.orm import session as se
from neutron_lib._i18n import _
from neutron_lib.db import constants as db_const
from neutron_lib.db import model_base
from neutron_lib.db import sqlalchemytypes
@event.listens_for(se.Session, 'after_bulk_delete')
def throw_exception_on_bulk_delete_of_listened_for_objects(delete_context):
    if hasattr(delete_context.mapper.class_, 'revises_on_change'):
        raise RuntimeError(_('%s may not be deleted in bulk because it bumps the revision of other resources via SQLAlchemy event handlers, which are not compatible with bulk deletes.') % delete_context.mapper.class_)