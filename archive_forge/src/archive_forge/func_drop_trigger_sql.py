from peewee import *
from playhouse.sqlite_ext import JSONField
def drop_trigger_sql(self, model, action):
    assert action in self._actions
    return self.drop_template % {'table': model._meta.table_name, 'action': action}