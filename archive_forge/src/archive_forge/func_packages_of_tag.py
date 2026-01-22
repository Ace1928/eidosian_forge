import pickle
import re
from debian.deprecation import function_deprecated_by
def packages_of_tag(self, tag):
    """Return the package set of a tag"""
    return self.rdb[tag] if tag in self.rdb else set()