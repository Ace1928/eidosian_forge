from pathlib import Path
from asgiref.local import Local
from django.apps import apps
from django.utils.autoreload import is_django_module
Clear the internal translations cache if a .mo file is modified.