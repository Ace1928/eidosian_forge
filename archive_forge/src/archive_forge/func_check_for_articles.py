from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def check_for_articles(self, heading, section):
    """Raise violation if the section begins with an article.

    See go/cloud-sdk-help-text#formatting.

    Arguments:
      heading: str, the name of the section.
      section: str, the contents of the section.

    Returns:
      True if there was a violation. False otherwise.
    """
    check_name = self._check_name(heading, 'ARTICLES')
    first_word = section.split()[0]
    if first_word.lower() in self._ARTICLES:
        self._add_failure(check_name, 'Please do not start the {} section with an article.'.format(heading))
        found_article = True
    else:
        self._add_success(check_name)
        found_article = False
    return found_article