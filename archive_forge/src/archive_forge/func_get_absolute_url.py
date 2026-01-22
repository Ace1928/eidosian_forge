from django.contrib.sites.models import Site
from django.db import models
from django.urls import NoReverseMatch, get_script_prefix, reverse
from django.utils.encoding import iri_to_uri
from django.utils.translation import gettext_lazy as _
def get_absolute_url(self):
    from .views import flatpage
    for url in (self.url.lstrip('/'), self.url):
        try:
            return reverse(flatpage, kwargs={'url': url})
        except NoReverseMatch:
            pass
    return iri_to_uri(get_script_prefix().rstrip('/') + self.url)