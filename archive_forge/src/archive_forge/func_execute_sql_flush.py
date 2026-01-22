import datetime
import decimal
import json
import warnings
from importlib import import_module
import sqlparse
from django.conf import settings
from django.db import NotSupportedError, transaction
from django.db.backends import utils
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import force_str
def execute_sql_flush(self, sql_list):
    """Execute a list of SQL statements to flush the database."""
    with transaction.atomic(using=self.connection.alias, savepoint=self.connection.features.can_rollback_ddl):
        with self.connection.cursor() as cursor:
            for sql in sql_list:
                cursor.execute(sql)