from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
@property
def sql_rename_column(self):
    is_mariadb = self.connection.mysql_is_mariadb
    if is_mariadb and self.connection.mysql_version < (10, 5, 2):
        return 'ALTER TABLE %(table)s CHANGE %(old_column)s %(new_column)s %(type)s'
    return super().sql_rename_column