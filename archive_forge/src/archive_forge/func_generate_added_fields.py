import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def generate_added_fields(self):
    """Make AddField operations."""
    for app_label, model_name, field_name in sorted(self.new_field_keys - self.old_field_keys):
        self._generate_added_field(app_label, model_name, field_name)