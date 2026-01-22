import functools
from keystone import exception
from keystone.i18n import _
def get_exact_filter_by_name(self, name):
    """Return a filter key and value if exact filter exists for name."""
    for entry in self.filters:
        if entry['name'] == name and entry['comparator'] == 'equals':
            return entry