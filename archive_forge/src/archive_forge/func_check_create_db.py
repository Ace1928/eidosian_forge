from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.spanner import database_operations
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def check_create_db(appname, instance_ref, database_id):
    """Create the DB if it doesn't exist already, raise otherwise."""
    schema_file = samples.get_local_schema_path(appname)
    database_dialect = samples.get_database_dialect(appname)
    schema = files.ReadFileContents(schema_file)
    if database_dialect == databases.DATABASE_DIALECT_POSTGRESQL:
        create_ddl = []
        schema = '\n'.join([line for line in schema.split('\n') if not line.startswith('--')])
        update_ddl = [stmt for stmt in schema.split(';') if stmt]
    else:
        create_ddl = ddl_parser.PreprocessDDLWithParser(schema)
        update_ddl = []
    create_op = _create_db_op(instance_ref, database_id, create_ddl, database_dialect)
    database_operations.Await(create_op, "Creating database '{}'".format(database_id))
    if update_ddl:
        database_ref = resources.REGISTRY.Parse(database_id, params={'instancesId': instance_ref.instancesId, 'projectsId': instance_ref.projectsId}, collection='spanner.projects.instances.databases')
        update_op = databases.UpdateDdl(database_ref, update_ddl)
        database_operations.Await(update_op, "Updating database '{}'".format(database_id))