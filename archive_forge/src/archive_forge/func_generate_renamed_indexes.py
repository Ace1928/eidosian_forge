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
def generate_renamed_indexes(self):
    for (app_label, model_name), alt_indexes in self.altered_indexes.items():
        for old_index_name, new_index_name, old_fields in alt_indexes['renamed_indexes']:
            self.add_operation(app_label, operations.RenameIndex(model_name=model_name, new_name=new_index_name, old_name=old_index_name, old_fields=old_fields))