from __future__ import unicode_literals
from collections import OrderedDict
import yaml
from pybtex.database.output import BaseWriter
def process_persons(persons):
    for person in persons:
        yield OrderedDict(process_person(person))