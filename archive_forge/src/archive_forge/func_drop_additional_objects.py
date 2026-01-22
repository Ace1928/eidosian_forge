import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
def drop_additional_objects(self, conn):
    enums = [e['name'] for e in sqlalchemy.inspect(conn).get_enums()]
    for e in enums:
        conn.exec_driver_sql('DROP TYPE %s' % e)