import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def test_db_signature(self):
    """
        Return a tuple with elements of self.connection.settings_dict (a
        DATABASES setting value) that uniquely identify a database
        accordingly to the RDBMS particularities.
        """
    settings_dict = self.connection.settings_dict
    return (settings_dict['HOST'], settings_dict['PORT'], settings_dict['ENGINE'], self._get_test_db_name())