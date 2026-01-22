import io
import sys
from typing import Dict, Any, Set
from pathlib import Path
from flask import Flask, render_template, request
from ase.db import connect
from ase.db.core import Database
from ase.formula import Formula
from ase.db.web import create_key_descriptions, Session
from ase.db.row import row2dct, AtomsRow
from ase.db.table import all_columns
@app.route('/robots.txt')
def robots():
    return ('User-agent: *\nDisallow: /\n\nUser-agent: Baiduspider\nDisallow: /\n\nUser-agent: SiteCheck-sitecrawl by Siteimprove.com\nDisallow: /\n', 200)