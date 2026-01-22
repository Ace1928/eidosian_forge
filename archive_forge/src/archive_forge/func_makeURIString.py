from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def makeURIString(self, template):
    """
        Replace the string "HOST" in C{template} with this test's host.

        Byte strings Python between (and including) versions 3.0 and 3.4
        cannot be formatted using C{%} or C{format} so this does a simple
        replace.

        @type template: L{bytes}
        @param template: A string containing "HOST".

        @rtype: L{bytes}
        @return: A string where "HOST" has been replaced by C{self.host}.
        """
    self.assertIsInstance(self.host, bytes)
    self.assertIsInstance(self.uriHost, bytes)
    self.assertIsInstance(template, bytes)
    self.assertIn(b'HOST', template)
    return template.replace(b'HOST', self.uriHost)