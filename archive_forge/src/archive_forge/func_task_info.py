import re
from collections import Counter
from fileinput import FileInput
import click
from celery.bin.base import CeleryCommand, handle_preload_options
def task_info(line):
    m = RE_TASK_INFO.match(line)
    return m.groups()