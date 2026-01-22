import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
@resource_status_reason.setter
def resource_status_reason(self, reason):
    self._resource_status_reason = reason and reason[:255] or ''