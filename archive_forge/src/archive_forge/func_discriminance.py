import pickle
import re
from debian.deprecation import function_deprecated_by
def discriminance(self, tag):
    """
        Return the discriminance index if the tag.

        Th discriminance index of the tag is defined as the minimum
        number of packages that would be eliminated by selecting only
        those tagged with this tag or only those not tagged with this
        tag.
        """
    n = self.card(tag)
    tot = self.package_count()
    return min(n, tot - n)