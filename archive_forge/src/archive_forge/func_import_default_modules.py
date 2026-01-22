import importlib
import os
import re
import sys
from datetime import datetime, timezone
from kombu.utils import json
from kombu.utils.objects import cached_property
from celery import signals
from celery.exceptions import reraise
from celery.utils.collections import DictAttribute, force_mapping
from celery.utils.functional import maybe_list
from celery.utils.imports import NotAPackage, find_module, import_from_cwd, symbol_by_name
def import_default_modules(self):
    responses = signals.import_modules.send(sender=self.app)
    for _, response in responses:
        if isinstance(response, Exception):
            raise response
    return [self.import_task_module(m) for m in self.default_modules]