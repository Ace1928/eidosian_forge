from inspect import getattr_static, unwrap
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ImproperlyConfigured, ObjectDoesNotExist
from django.http import Http404, HttpResponse
from django.template import TemplateDoesNotExist, loader
from django.utils import feedgenerator
from django.utils.encoding import iri_to_uri
from django.utils.html import escape
from django.utils.http import http_date
from django.utils.timezone import get_default_timezone, is_naive, make_aware
from django.utils.translation import get_language
def item_link(self, item):
    try:
        return item.get_absolute_url()
    except AttributeError:
        raise ImproperlyConfigured('Give your %s class a get_absolute_url() method, or define an item_link() method in your Feed class.' % item.__class__.__name__)