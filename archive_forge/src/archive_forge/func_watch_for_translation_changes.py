from pathlib import Path
from asgiref.local import Local
from django.apps import apps
from django.utils.autoreload import is_django_module
def watch_for_translation_changes(sender, **kwargs):
    """Register file watchers for .mo files in potential locale paths."""
    from django.conf import settings
    if settings.USE_I18N:
        directories = [Path('locale')]
        directories.extend((Path(config.path) / 'locale' for config in apps.get_app_configs() if not is_django_module(config.module)))
        directories.extend((Path(p) for p in settings.LOCALE_PATHS))
        for path in directories:
            sender.watch_dir(path, '**/*.mo')