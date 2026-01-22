import re
from .pandoc import convert_pandoc
def markdown2html_mistune(source: str) -> str:
    """mistune is unavailable, raise ImportError"""
    msg = f'markdown2html requires mistune: {_mistune_import_error}'
    raise ImportError(msg)