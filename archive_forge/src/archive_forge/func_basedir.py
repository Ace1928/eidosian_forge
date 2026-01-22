import os
import re
from importlib import import_module
from django import get_version
from django.apps import apps
from django.conf import SettingsReference  # NOQA
from django.db import migrations
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.serializer import Serializer, serializer_factory
from django.utils.inspect import get_func_args
from django.utils.module_loading import module_dir
from django.utils.timezone import now
@property
def basedir(self):
    migrations_package_name, _ = MigrationLoader.migrations_module(self.migration.app_label)
    if migrations_package_name is None:
        raise ValueError("Django can't create migrations for app '%s' because migrations have been disabled via the MIGRATION_MODULES setting." % self.migration.app_label)
    try:
        migrations_module = import_module(migrations_package_name)
    except ImportError:
        pass
    else:
        try:
            return module_dir(migrations_module)
        except ValueError:
            pass
    app_config = apps.get_app_config(self.migration.app_label)
    maybe_app_name, _, migrations_package_basename = migrations_package_name.rpartition('.')
    if app_config.name == maybe_app_name:
        return os.path.join(app_config.path, migrations_package_basename)
    existing_dirs, missing_dirs = (migrations_package_name.split('.'), [])
    while existing_dirs:
        missing_dirs.insert(0, existing_dirs.pop(-1))
        try:
            base_module = import_module('.'.join(existing_dirs))
        except (ImportError, ValueError):
            continue
        else:
            try:
                base_dir = module_dir(base_module)
            except ValueError:
                continue
            else:
                break
    else:
        raise ValueError('Could not locate an appropriate location to create migrations package %s. Make sure the toplevel package exists and can be imported.' % migrations_package_name)
    final_dir = os.path.join(base_dir, *missing_dirs)
    os.makedirs(final_dir, exist_ok=True)
    for missing_dir in missing_dirs:
        base_dir = os.path.join(base_dir, missing_dir)
        with open(os.path.join(base_dir, '__init__.py'), 'w'):
            pass
    return final_dir