import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def postcurrent(self, next):
    """Postprocess the current element taking into account next and last."""
    stage = self.stages.getstage(self.current)
    if not stage:
        return self.current
    return stage.postprocess(self.last, self.current, next)