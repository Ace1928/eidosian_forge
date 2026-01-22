import copy
import datetime
from django.core.exceptions import NON_FIELD_ERRORS, ValidationError
from django.forms.fields import Field, FileField
from django.forms.utils import ErrorDict, ErrorList, RenderableFormMixin
from django.forms.widgets import Media, MediaDefiningClass
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from .renderers import get_default_renderer
def order_fields(self, field_order):
    """
        Rearrange the fields according to field_order.

        field_order is a list of field names specifying the order. Append fields
        not included in the list in the default order for backward compatibility
        with subclasses not overriding field_order. If field_order is None,
        keep all fields in the order defined in the class. Ignore unknown
        fields in field_order to allow disabling fields in form subclasses
        without redefining ordering.
        """
    if field_order is None:
        return
    fields = {}
    for key in field_order:
        try:
            fields[key] = self.fields.pop(key)
        except KeyError:
            pass
    fields.update(self.fields)
    self.fields = fields