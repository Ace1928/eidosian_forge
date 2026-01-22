import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def get_test_db_clone_settings(self, suffix):
    """
        Return a modified connection settings dict for the n-th clone of a DB.
        """
    orig_settings_dict = self.connection.settings_dict
    return {**orig_settings_dict, 'NAME': '{}_{}'.format(orig_settings_dict['NAME'], suffix)}