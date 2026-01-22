import os
import select
import sys
import traceback
from django.core.management import BaseCommand, CommandError
from django.utils.datastructures import OrderedSet
def bpython(self, options):
    import bpython
    bpython.embed()