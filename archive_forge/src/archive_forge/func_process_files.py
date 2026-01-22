import glob
import os
import re
import sys
from functools import total_ordering
from itertools import dropwhile
from pathlib import Path
import django
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.files.temp import NamedTemporaryFile
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import (
from django.utils.encoding import DEFAULT_LOCALE_ENCODING
from django.utils.functional import cached_property
from django.utils.jslex import prepare_js_for_gettext
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import get_text_list
from django.utils.translation import templatize
def process_files(self, file_list):
    """
        Group translatable files by locale directory and run pot file build
        process for each group.
        """
    file_groups = {}
    for translatable in file_list:
        file_group = file_groups.setdefault(translatable.locale_dir, [])
        file_group.append(translatable)
    for locale_dir, files in file_groups.items():
        self.process_locale_dir(locale_dir, files)