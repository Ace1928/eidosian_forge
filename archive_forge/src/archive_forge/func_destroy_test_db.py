import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def destroy_test_db(self, old_database_name=None, verbosity=1, keepdb=False, suffix=None):
    """
        Destroy a test database, prompting the user for confirmation if the
        database already exists.
        """
    self.connection.close()
    if suffix is None:
        test_database_name = self.connection.settings_dict['NAME']
    else:
        test_database_name = self.get_test_db_clone_settings(suffix)['NAME']
    if verbosity >= 1:
        action = 'Destroying'
        if keepdb:
            action = 'Preserving'
        self.log('%s test database for alias %s...' % (action, self._get_database_display_str(verbosity, test_database_name)))
    if not keepdb:
        self._destroy_test_db(test_database_name, verbosity)
    if old_database_name is not None:
        settings.DATABASES[self.connection.alias]['NAME'] = old_database_name
        self.connection.settings_dict['NAME'] = old_database_name