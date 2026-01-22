import pickle
import re
from debian.deprecation import function_deprecated_by
def tags_of_packages(self, pkgs):
    """Return the set of tags that have all the packages in ``pkgs``"""
    return set.union(*(self.tags_of_package(p) for p in pkgs))