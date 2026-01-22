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
@declarative.declared_attr
def standard_attr_id(cls):
    return sa.Column(sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), sa.ForeignKey(StandardAttribute.id, ondelete='CASCADE'), unique=True, nullable=False)