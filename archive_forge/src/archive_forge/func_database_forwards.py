from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
def database_forwards(self, app_label, schema_editor, from_state, to_state):
    model = from_state.apps.get_model(app_label, self.model_name)
    if self.allow_migrate_model(schema_editor.connection.alias, model):
        schema_editor.execute('ALTER TABLE %s VALIDATE CONSTRAINT %s' % (schema_editor.quote_name(model._meta.db_table), schema_editor.quote_name(self.name)))