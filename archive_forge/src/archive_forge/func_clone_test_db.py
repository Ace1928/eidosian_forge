import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def clone_test_db(self, suffix, verbosity=1, autoclobber=False, keepdb=False):
    """
        Clone a test database.
        """
    source_database_name = self.connection.settings_dict['NAME']
    if verbosity >= 1:
        action = 'Cloning test database'
        if keepdb:
            action = 'Using existing clone'
        self.log('%s for alias %s...' % (action, self._get_database_display_str(verbosity, source_database_name)))
    self._clone_test_db(suffix, verbosity, keepdb)