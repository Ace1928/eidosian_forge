import itertools
from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.core.management import BaseCommand
from django.db import DEFAULT_DB_ALIAS, router
from django.db.models.deletion import Collector

        Always load related objects to display them when showing confirmation.
        