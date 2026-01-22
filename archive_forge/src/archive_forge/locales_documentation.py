import gettext
import logging
import os
import sqlite3
import sys

    Get the human readable and translated name of a language based on it's code.

    :param code: the code of the language (e.g. de_DE, en_US)
    :param target: separator used to separate language from country
    :rtype: human readable and translated language name
    