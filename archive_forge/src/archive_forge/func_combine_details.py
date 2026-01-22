import itertools
import sys
from fixtures.callmany import (
def combine_details(source_details, target_details):
    """Add every value from source to target deduping common keys."""
    for name, content_object in source_details.items():
        new_name = name
        disambiguator = itertools.count(1)
        while new_name in target_details:
            new_name = '%s-%d' % (name, next(disambiguator))
        name = new_name
        target_details[name] = content_object