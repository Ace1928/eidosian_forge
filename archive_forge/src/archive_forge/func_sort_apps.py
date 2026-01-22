import html
import re
import os
from collections.abc import MutableMapping as DictMixin
from paste import httpexceptions
def sort_apps(self):
    """
        Make sure applications are sorted with longest URLs first
        """

    def key(app_desc):
        (domain, url), app = app_desc
        if not domain:
            return ('ÿ', -len(url))
        else:
            return (domain, -len(url))
    apps = [(key(desc), desc) for desc in self.applications]
    apps.sort()
    self.applications = [desc for sortable, desc in apps]