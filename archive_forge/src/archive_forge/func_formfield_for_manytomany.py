from django.conf import settings
from django.contrib import admin, messages
from django.contrib.admin.options import IS_POPUP_VAR
from django.contrib.admin.utils import unquote
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import (
from django.contrib.auth.models import Group, User
from django.core.exceptions import PermissionDenied
from django.db import router, transaction
from django.http import Http404, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.decorators import method_decorator
from django.utils.html import escape
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
def formfield_for_manytomany(self, db_field, request=None, **kwargs):
    if db_field.name == 'permissions':
        qs = kwargs.get('queryset', db_field.remote_field.model.objects)
        kwargs['queryset'] = qs.select_related('content_type')
    return super().formfield_for_manytomany(db_field, request=request, **kwargs)