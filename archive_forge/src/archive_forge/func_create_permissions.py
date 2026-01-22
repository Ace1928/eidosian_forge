import getpass
import unicodedata
from django.apps import apps as global_apps
from django.contrib.auth import get_permission_codename
from django.contrib.contenttypes.management import create_contenttypes
from django.core import exceptions
from django.db import DEFAULT_DB_ALIAS, router
def create_permissions(app_config, verbosity=2, interactive=True, using=DEFAULT_DB_ALIAS, apps=global_apps, **kwargs):
    if not app_config.models_module:
        return
    create_contenttypes(app_config, verbosity=verbosity, interactive=interactive, using=using, apps=apps, **kwargs)
    app_label = app_config.label
    try:
        app_config = apps.get_app_config(app_label)
        ContentType = apps.get_model('contenttypes', 'ContentType')
        Permission = apps.get_model('auth', 'Permission')
    except LookupError:
        return
    if not router.allow_migrate_model(using, Permission):
        return
    searched_perms = []
    ctypes = set()
    for klass in app_config.get_models():
        ctype = ContentType.objects.db_manager(using).get_for_model(klass, for_concrete_model=False)
        ctypes.add(ctype)
        for perm in _get_all_permissions(klass._meta):
            searched_perms.append((ctype, perm))
    all_perms = set(Permission.objects.using(using).filter(content_type__in=ctypes).values_list('content_type', 'codename'))
    perms = []
    for ct, (codename, name) in searched_perms:
        if (ct.pk, codename) not in all_perms:
            permission = Permission()
            permission._state.db = using
            permission.codename = codename
            permission.name = name
            permission.content_type = ct
            perms.append(permission)
    Permission.objects.using(using).bulk_create(perms)
    if verbosity >= 2:
        for perm in perms:
            print("Adding permission '%s'" % perm)