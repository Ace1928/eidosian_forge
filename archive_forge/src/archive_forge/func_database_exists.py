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
def database_exists(self, engine, ident):
    return bool(engine.scalar(sqlalchemy.text('SELECT datname FROM pg_database WHERE datname=:name'), {'name': ident}))