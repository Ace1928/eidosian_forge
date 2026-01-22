from __future__ import annotations
import json
import re
from traitlets.log import get_logger
from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
from .nbbase import nbformat, nbformat_minor
def upgrade_outputs(outputs):
    """upgrade outputs of a code cell from v3 to v4"""
    return [upgrade_output(op) for op in outputs]