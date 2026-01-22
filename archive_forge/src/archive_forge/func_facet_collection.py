import pickle
import re
from debian.deprecation import function_deprecated_by
def facet_collection(self):
    """
        Return a copy of this collection, but replaces the tag names
        with only their facets.
        """
    fcoll = DB()
    tofacet = re.compile('^([^:]+).+')
    for pkg, tags in self.iter_packages_tags():
        ftags = {tofacet.sub('\\1', t) for t in tags}
        fcoll.insert(pkg, ftags)
    return fcoll