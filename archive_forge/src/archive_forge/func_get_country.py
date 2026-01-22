import gettext
import logging
import os
import sqlite3
import sys
@classmethod
def get_country(cls, code, codec):
    country = _database.execute('SELECT rowid FROM countries WHERE %s == ?' % codec, (code,)).fetchone()
    if country:
        return cls(country[0])
    raise CountryNotFound('code: %s, codec: %s' % (code, codec))