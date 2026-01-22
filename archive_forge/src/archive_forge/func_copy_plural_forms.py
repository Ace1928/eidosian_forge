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
def copy_plural_forms(self, msgs, locale):
    """
        Copy plural forms header contents from a Django catalog of locale to
        the msgs string, inserting it at the right place. msgs should be the
        contents of a newly created .po file.
        """
    django_dir = os.path.normpath(os.path.join(os.path.dirname(django.__file__)))
    if self.domain == 'djangojs':
        domains = ('djangojs', 'django')
    else:
        domains = ('django',)
    for domain in domains:
        django_po = os.path.join(django_dir, 'conf', 'locale', locale, 'LC_MESSAGES', '%s.po' % domain)
        if os.path.exists(django_po):
            with open(django_po, encoding='utf-8') as fp:
                m = plural_forms_re.search(fp.read())
            if m:
                plural_form_line = m['value']
                if self.verbosity > 1:
                    self.stdout.write('copying plural forms: %s' % plural_form_line)
                lines = []
                found = False
                for line in msgs.splitlines():
                    if not found and (not line or plural_forms_re.search(line)):
                        line = plural_form_line
                        found = True
                    lines.append(line)
                msgs = '\n'.join(lines)
                break
    return msgs