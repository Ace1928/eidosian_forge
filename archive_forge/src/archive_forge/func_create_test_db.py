import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def create_test_db(self, verbosity=1, autoclobber=False, serialize=True, keepdb=False):
    """
        Create a test database, prompting the user for confirmation if the
        database already exists. Return the name of the test database created.
        """
    from django.core.management import call_command
    test_database_name = self._get_test_db_name()
    if verbosity >= 1:
        action = 'Creating'
        if keepdb:
            action = 'Using existing'
        self.log('%s test database for alias %s...' % (action, self._get_database_display_str(verbosity, test_database_name)))
    self._create_test_db(verbosity, autoclobber, keepdb)
    self.connection.close()
    settings.DATABASES[self.connection.alias]['NAME'] = test_database_name
    self.connection.settings_dict['NAME'] = test_database_name
    try:
        if self.connection.settings_dict['TEST']['MIGRATE'] is False:
            old_migration_modules = settings.MIGRATION_MODULES
            settings.MIGRATION_MODULES = {app.label: None for app in apps.get_app_configs()}
        call_command('migrate', verbosity=max(verbosity - 1, 0), interactive=False, database=self.connection.alias, run_syncdb=True)
    finally:
        if self.connection.settings_dict['TEST']['MIGRATE'] is False:
            settings.MIGRATION_MODULES = old_migration_modules
    if serialize:
        self.connection._test_serialized_contents = self.serialize_db_to_string()
    call_command('createcachetable', database=self.connection.alias)
    self.connection.ensure_connection()
    if os.environ.get('RUNNING_DJANGOS_TEST_SUITE') == 'true':
        self.mark_expected_failures_and_skips()
    return test_database_name