from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
def create_collation(self, schema_editor):
    args = {'locale': schema_editor.quote_name(self.locale)}
    if self.provider != 'libc':
        args['provider'] = schema_editor.quote_name(self.provider)
    if self.deterministic is False:
        args['deterministic'] = 'false'
    schema_editor.execute('CREATE COLLATION %(name)s (%(args)s)' % {'name': schema_editor.quote_name(self.name), 'args': ', '.join((f'{option}={value}' for option, value in args.items()))})