import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def replacespecial(self, line):
    """Replace all special chars from elyxer.a line"""
    replaced = self.escape(line, EscapeConfig.entities)
    replaced = self.changeline(replaced)
    if ContainerConfig.string['startcommand'] in replaced and len(replaced) > 1:
        if self.begin:
            message = 'Unknown command at ' + str(self.begin) + ': '
        else:
            message = 'Unknown command: '
        Trace.error(message + replaced.strip())
    return replaced