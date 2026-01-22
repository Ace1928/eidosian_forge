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
def build_potfiles(self):
    """
        Build pot files and apply msguniq to them.
        """
    file_list = self.find_files('.')
    self.remove_potfiles()
    self.process_files(file_list)
    potfiles = []
    for path in self.locale_paths:
        potfile = os.path.join(path, '%s.pot' % self.domain)
        if not os.path.exists(potfile):
            continue
        args = ['msguniq'] + self.msguniq_options + [potfile]
        msgs, errors, status = popen_wrapper(args)
        if errors:
            if status != STATUS_OK:
                raise CommandError('errors happened while running msguniq\n%s' % errors)
            elif self.verbosity > 0:
                self.stdout.write(errors)
        msgs = normalize_eols(msgs)
        with open(potfile, 'w', encoding='utf-8') as fp:
            fp.write(msgs)
        potfiles.append(potfile)
    return potfiles