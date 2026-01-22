from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
@property
def cleaned_data(self):
    """
        Return a list of form.cleaned_data dicts for every form in self.forms.
        """
    if not self.is_valid():
        raise AttributeError("'%s' object has no attribute 'cleaned_data'" % self.__class__.__name__)
    return [form.cleaned_data for form in self.forms]