import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
def get_studies(base_dir=None, followlinks=False):
    if base_dir is not None:
        update_cache(base_dir, followlinks)
    if base_dir is None:
        with DB.readonly_cursor() as c:
            c.execute('SELECT * FROM study')
            studies = []
            cols = [el[0] for el in c.description]
            for row in c:
                d = dict(zip(cols, row))
                studies.append(_Study(d))
        return studies
    query = 'SELECT study\n                 FROM series\n                WHERE uid IN (SELECT series\n                                FROM storage_instance\n                               WHERE uid IN (SELECT storage_instance\n                                               FROM file\n                                              WHERE directory = ?))'
    with DB.readonly_cursor() as c:
        study_uids = {}
        for dir in _get_subdirs(base_dir, followlinks=followlinks):
            c.execute(query, (dir,))
            for row in c:
                study_uids[row[0]] = None
        studies = []
        for uid in study_uids:
            c.execute('SELECT * FROM study WHERE uid = ?', (uid,))
            cols = [el[0] for el in c.description]
            d = dict(zip(cols, c.fetchone()))
            studies.append(_Study(d))
    return studies