from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
def get_field_parameters(self):
    params = {}
    if self.extra_parameters is not None:
        params.update(self.extra_parameters)
    if self.nullable:
        params['null'] = True
    if self.field_class is ForeignKeyField or self.name != self.column_name:
        params['column_name'] = "'%s'" % self.column_name
    if self.primary_key and (not issubclass(self.field_class, AutoField)):
        params['primary_key'] = True
    if self.default is not None:
        params['constraints'] = '[SQL("DEFAULT %s")]' % self.default
    if self.is_foreign_key():
        params['model'] = self.rel_model
        if self.to_field:
            params['field'] = "'%s'" % self.to_field
        if self.related_name:
            params['backref'] = "'%s'" % self.related_name
    if not self.is_primary_key():
        if self.unique:
            params['unique'] = 'True'
        elif self.index and (not self.is_foreign_key()):
            params['index'] = 'True'
    return params