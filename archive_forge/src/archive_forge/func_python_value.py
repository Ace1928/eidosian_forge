import json
from peewee import ImproperlyConfigured
from peewee import Insert
from peewee import MySQLDatabase
from peewee import Node
from peewee import NodeList
from peewee import SQL
from peewee import TextField
from peewee import fn
from peewee import __deprecated__
def python_value(self, value):
    if value is not None:
        try:
            return self._json_loads(value)
        except (TypeError, ValueError):
            return value