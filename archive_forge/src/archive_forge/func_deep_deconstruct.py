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
def deep_deconstruct(self, obj):
    """
        Recursive deconstruction for a field and its arguments.
        Used for full comparison for rename/alter; sometimes a single-level
        deconstruction will not compare correctly.
        """
    if isinstance(obj, list):
        return [self.deep_deconstruct(value) for value in obj]
    elif isinstance(obj, tuple):
        return tuple((self.deep_deconstruct(value) for value in obj))
    elif isinstance(obj, dict):
        return {key: self.deep_deconstruct(value) for key, value in obj.items()}
    elif isinstance(obj, functools.partial):
        return (obj.func, self.deep_deconstruct(obj.args), self.deep_deconstruct(obj.keywords))
    elif isinstance(obj, COMPILED_REGEX_TYPE):
        return RegexObject(obj)
    elif isinstance(obj, type):
        return obj
    elif hasattr(obj, 'deconstruct'):
        deconstructed = obj.deconstruct()
        if isinstance(obj, models.Field):
            deconstructed = deconstructed[1:]
        path, args, kwargs = deconstructed
        return (path, [self.deep_deconstruct(value) for value in args], {key: self.deep_deconstruct(value) for key, value in kwargs.items()})
    else:
        return obj