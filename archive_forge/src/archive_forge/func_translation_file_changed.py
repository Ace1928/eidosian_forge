from pathlib import Path
from asgiref.local import Local
from django.apps import apps
from django.utils.autoreload import is_django_module
def translation_file_changed(sender, file_path, **kwargs):
    """Clear the internal translations cache if a .mo file is modified."""
    if file_path.suffix == '.mo':
        import gettext
        from django.utils.translation import trans_real
        gettext._translations = {}
        trans_real._translations = {}
        trans_real._default = None
        trans_real._active = Local()
        return True