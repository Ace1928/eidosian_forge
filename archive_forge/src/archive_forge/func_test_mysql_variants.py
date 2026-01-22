from sqlalchemy import Column, Integer
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import declarative_base
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import types
from oslo_db.tests.sqlalchemy import base as test_base
def test_mysql_variants(self):
    self.assertEqual('LONGTEXT', str(types.JsonEncodedDict(mysql_as_long=True).compile(dialect=mysql.dialect())))
    self.assertEqual('MEDIUMTEXT', str(types.JsonEncodedDict(mysql_as_medium=True).compile(dialect=mysql.dialect())))
    self.assertRaises(TypeError, lambda: types.JsonEncodedDict(mysql_as_long=True, mysql_as_medium=True))