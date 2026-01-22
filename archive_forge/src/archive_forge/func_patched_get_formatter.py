import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def patched_get_formatter(self, i, *args, **kwargs):
    if isinstance(self.formatters, dict) and isinstance(i, int) and (i not in self.columns) and hasattr(self, 'tr_col_num') and (i >= self.tr_col_num):
        max_cols = 0
        if hasattr(self, 'max_cols_fitted'):
            max_cols = self.max_cols_fitted
        elif hasattr(self, 'max_cols_adj'):
            max_cols = self.max_cols_adj
        n_trunc_cols = len(self.columns) - max_cols
        if n_trunc_cols > 0:
            i += n_trunc_cols
    return orig_get_formatter(self, i, *args, **kwargs)