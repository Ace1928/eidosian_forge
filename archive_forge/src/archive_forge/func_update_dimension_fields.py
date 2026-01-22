import datetime
import posixpath
from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.core.files.utils import validate_file_name
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData
from django.utils.translation import gettext_lazy as _
def update_dimension_fields(self, instance, force=False, *args, **kwargs):
    """
        Update field's width and height fields, if defined.

        This method is hooked up to model's post_init signal to update
        dimensions after instantiating a model instance.  However, dimensions
        won't be updated if the dimensions fields are already populated.  This
        avoids unnecessary recalculation when loading an object from the
        database.

        Dimensions can be forced to update with force=True, which is how
        ImageFileDescriptor.__set__ calls this method.
        """
    has_dimension_fields = self.width_field or self.height_field
    if not has_dimension_fields or self.attname not in instance.__dict__:
        return
    file = getattr(instance, self.attname)
    if not file and (not force):
        return
    dimension_fields_filled = not (self.width_field and (not getattr(instance, self.width_field)) or (self.height_field and (not getattr(instance, self.height_field))))
    if dimension_fields_filled and (not force):
        return
    if file:
        width = file.width
        height = file.height
    else:
        width = None
        height = None
    if self.width_field:
        setattr(instance, self.width_field, width)
    if self.height_field:
        setattr(instance, self.height_field, height)