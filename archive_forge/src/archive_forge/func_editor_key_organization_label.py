from __future__ import absolute_import, unicode_literals
import re
import sys
import unicodedata
import six
from pybtex.style.labels import BaseLabelStyle
from pybtex.textutils import abbreviate
def editor_key_organization_label(self, entry):
    if not 'editor' in entry.persons:
        if not 'key' in entry.fields:
            if not 'organization' in entry.fields:
                return entry.key[:3]
            else:
                result = entry.fields['organization']
                if result.startswith('The '):
                    result = result[4:]
                return result
        else:
            return entry.fields['key'][:3]
    else:
        return self.format_lab_names(entry.persons['editor'])