import re
from collections import Counter
from fileinput import FileInput
import click
from celery.bin.base import CeleryCommand, handle_preload_options
def task_received(self, line, task_name, task_id):
    self.names[task_id] = task_name
    self.ids.add(task_id)
    self.task_types[task_name] += 1