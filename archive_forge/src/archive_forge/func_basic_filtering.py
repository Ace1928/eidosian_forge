from collections import namedtuple
import re
import textwrap
import warnings
def basic_filtering(self, language_tags):
    """
        Return the tags that match the header, using Basic Filtering.

        :param language_tags: (``iterable``) language tags
        :return: A list of tuples of the form (language tag, qvalue), in
                 descending order of preference.

        When the header is invalid and when the header is not in the request,
        there are no matches, so this method always returns an empty list.
        """
    return []