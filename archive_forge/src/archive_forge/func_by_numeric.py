import gettext
import logging
import os
import sqlite3
import sys
@classmethod
def by_numeric(cls, code):
    return Country.get_country(code, 'numeric')