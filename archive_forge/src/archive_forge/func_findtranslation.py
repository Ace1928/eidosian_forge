import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def findtranslation(self):
    """Find the translation for the document language."""
    self.langcodes = None
    if not DocumentParameters.language:
        Trace.error('No language in document')
        return
    if not DocumentParameters.language in TranslationConfig.languages:
        Trace.error('Unknown language ' + DocumentParameters.language)
        return
    if TranslationConfig.languages[DocumentParameters.language] == 'en':
        return
    langcodes = [TranslationConfig.languages[DocumentParameters.language]]
    try:
        self.translation = gettext.translation('elyxer', None, langcodes)
    except IOError:
        Trace.error('No translation for ' + str(langcodes))