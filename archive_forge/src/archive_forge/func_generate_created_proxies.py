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
def generate_created_proxies(self):
    """
        Make CreateModel statements for proxy models. Use the same statements
        as that way there's less code duplication, but for proxy models it's
        safe to skip all the pointless field stuff and chuck out an operation.
        """
    added = self.new_proxy_keys - self.old_proxy_keys
    for app_label, model_name in sorted(added):
        model_state = self.to_state.models[app_label, model_name]
        assert model_state.options.get('proxy')
        dependencies = [(app_label, model_name, None, False)]
        for base in model_state.bases:
            if isinstance(base, str) and '.' in base:
                base_app_label, base_name = base.split('.', 1)
                dependencies.append((base_app_label, base_name, None, True))
        self.add_operation(app_label, operations.CreateModel(name=model_state.name, fields=[], options=model_state.options, bases=model_state.bases, managers=model_state.managers), dependencies=dependencies)