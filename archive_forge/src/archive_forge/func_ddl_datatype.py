import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
def ddl_datatype(self, ctx):
    data_type = self.__field.ddl_datatype(ctx)
    return NodeList((data_type, SQL('[]' * self.dimensions)), glue='')