from itertools import chain
from django.core.exceptions import (
from django.db.models.utils import AltersData
from django.forms.fields import ChoiceField, Field
from django.forms.forms import BaseForm, DeclarativeFieldsMetaclass
from django.forms.formsets import BaseFormSet, formset_factory
from django.forms.utils import ErrorList
from django.forms.widgets import (
from django.utils.choices import BaseChoiceIterator
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def get_date_error_message(self, date_check):
    return gettext('Please correct the duplicate data for %(field_name)s which must be unique for the %(lookup)s in %(date_field)s.') % {'field_name': date_check[2], 'date_field': date_check[3], 'lookup': str(date_check[1])}