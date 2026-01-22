import pickle
import re
from debian.deprecation import function_deprecated_by
def iter_packages_tags(self):
    """Iterate over 2-tuples of (pkg, tags)"""
    return self.db.items()