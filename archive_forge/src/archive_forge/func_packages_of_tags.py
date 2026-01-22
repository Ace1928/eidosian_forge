import pickle
import re
from debian.deprecation import function_deprecated_by
def packages_of_tags(self, tags):
    """Return the set of packages that have all the tags in ``tags``"""
    return set.union(*(self.packages_of_tag(t) for t in tags))