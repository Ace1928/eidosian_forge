import getpass
import unicodedata
from django.apps import apps as global_apps
from django.contrib.auth import get_permission_codename
from django.contrib.contenttypes.management import create_contenttypes
from django.core import exceptions
from django.db import DEFAULT_DB_ALIAS, router
def get_default_username(check_db=True, database=DEFAULT_DB_ALIAS):
    """
    Try to determine the current system user's username to use as a default.

    :param check_db: If ``True``, requires that the username does not match an
        existing ``auth.User`` (otherwise returns an empty string).
    :param database: The database where the unique check will be performed.
    :returns: The username, or an empty string if no username can be
        determined or the suggested username is already taken.
    """
    from django.contrib.auth import models as auth_app
    if auth_app.User._meta.swapped:
        return ''
    default_username = get_system_username()
    try:
        default_username = unicodedata.normalize('NFKD', default_username).encode('ascii', 'ignore').decode('ascii').replace(' ', '').lower()
    except UnicodeDecodeError:
        return ''
    try:
        auth_app.User._meta.get_field('username').run_validators(default_username)
    except exceptions.ValidationError:
        return ''
    if check_db and default_username:
        try:
            auth_app.User._default_manager.db_manager(database).get(username=default_username)
        except auth_app.User.DoesNotExist:
            pass
        else:
            return ''
    return default_username