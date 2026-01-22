import json
from django import forms
from django.contrib.admin.utils import (
from django.core.exceptions import ObjectDoesNotExist
from django.db.models.fields.related import (
from django.forms.utils import flatatt
from django.template.defaultfilters import capfirst, linebreaksbr
from django.urls import NoReverseMatch, reverse
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def inline_formset_data(self):
    verbose_name = self.opts.verbose_name
    return json.dumps({'name': '#%s' % self.formset.prefix, 'options': {'prefix': self.formset.prefix, 'addText': gettext('Add another %(verbose_name)s') % {'verbose_name': capfirst(verbose_name)}, 'deleteText': gettext('Remove')}})