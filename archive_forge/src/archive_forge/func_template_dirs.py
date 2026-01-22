from django.core.exceptions import ImproperlyConfigured, SuspiciousFileOperation
from django.template.utils import get_app_template_dirs
from django.utils._os import safe_join
from django.utils.functional import cached_property
@cached_property
def template_dirs(self):
    """
        Return a list of directories to search for templates.
        """
    template_dirs = tuple(self.dirs)
    if self.app_dirs:
        template_dirs += get_app_template_dirs(self.app_dirname)
    return template_dirs