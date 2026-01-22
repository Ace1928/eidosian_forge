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
def create_opportunistic_driver_url(self):
    return 'postgresql+psycopg2://openstack_citest:openstack_citest@localhost/postgres'