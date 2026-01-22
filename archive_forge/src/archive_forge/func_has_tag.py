import pickle
import re
from debian.deprecation import function_deprecated_by
def has_tag(self, tag):
    """Check if the collection contains packages tagged with tag"""
    return tag in self.rdb