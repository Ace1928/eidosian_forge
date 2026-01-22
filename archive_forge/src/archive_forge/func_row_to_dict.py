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
def row_to_dict(row: AtomsRow, project: Dict[str, Any]) -> Dict[str, Any]:
    """Convert row to dict for use in html template."""
    dct = row2dct(row, project['key_descriptions'])
    dct['formula'] = Formula(Formula(row.formula).format('abc')).format('html')
    return dct