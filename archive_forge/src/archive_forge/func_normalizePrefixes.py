from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def normalizePrefixes(self):
    """
        Normalize the namespace prefixes.

        This generates unique prefixes for all namespaces. Then retrofits all
        prefixes and prefix mappings. Further, it will retrofix attribute
        values that have values containing (:).

        @return: self
        @rtype: L{Element}

        """
    PrefixNormalizer.apply(self)
    return self