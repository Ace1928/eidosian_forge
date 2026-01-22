import getpass
import unicodedata
from django.apps import apps as global_apps
from django.contrib.auth import get_permission_codename
from django.contrib.contenttypes.management import create_contenttypes
from django.core import exceptions
from django.db import DEFAULT_DB_ALIAS, router
def get_system_username():
    """
    Return the current system user's username, or an empty string if the
    username could not be determined.
    """
    try:
        result = getpass.getuser()
    except (ImportError, KeyError):
        return ''
    return result