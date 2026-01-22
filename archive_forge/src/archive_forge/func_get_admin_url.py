import json
from django.conf import settings
from django.contrib.admin.utils import quote
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.urls import NoReverseMatch, reverse
from django.utils import timezone
from django.utils.text import get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def get_admin_url(self):
    """
        Return the admin URL to edit the object represented by this log entry.
        """
    if self.content_type and self.object_id:
        url_name = 'admin:%s_%s_change' % (self.content_type.app_label, self.content_type.model)
        try:
            return reverse(url_name, args=(quote(self.object_id),))
        except NoReverseMatch:
            pass
    return None