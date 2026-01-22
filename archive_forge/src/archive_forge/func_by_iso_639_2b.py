import gettext
import logging
import os
import sqlite3
import sys
@classmethod
def by_iso_639_2b(cls, code):
    return Language.get_language(code, 'iso_639_2B')