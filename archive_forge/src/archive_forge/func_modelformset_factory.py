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
def modelformset_factory(model, form=ModelForm, formfield_callback=None, formset=BaseModelFormSet, extra=1, can_delete=False, can_order=False, max_num=None, fields=None, exclude=None, widgets=None, validate_max=False, localized_fields=None, labels=None, help_texts=None, error_messages=None, min_num=None, validate_min=False, field_classes=None, absolute_max=None, can_delete_extra=True, renderer=None, edit_only=False):
    """Return a FormSet class for the given Django model class."""
    meta = getattr(form, 'Meta', None)
    if getattr(meta, 'fields', fields) is None and getattr(meta, 'exclude', exclude) is None:
        raise ImproperlyConfigured("Calling modelformset_factory without defining 'fields' or 'exclude' explicitly is prohibited.")
    form = modelform_factory(model, form=form, fields=fields, exclude=exclude, formfield_callback=formfield_callback, widgets=widgets, localized_fields=localized_fields, labels=labels, help_texts=help_texts, error_messages=error_messages, field_classes=field_classes)
    FormSet = formset_factory(form, formset, extra=extra, min_num=min_num, max_num=max_num, can_order=can_order, can_delete=can_delete, validate_min=validate_min, validate_max=validate_max, absolute_max=absolute_max, can_delete_extra=can_delete_extra, renderer=renderer)
    FormSet.model = model
    FormSet.edit_only = edit_only
    return FormSet