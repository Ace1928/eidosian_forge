import re
from ..helpers import PREVENT_BACKSLASH
def table_in_quote(md):
    """Enable table plugin in block quotes."""
    md.block.insert_rule(md.block.block_quote_rules, 'table', before='paragraph')
    md.block.insert_rule(md.block.block_quote_rules, 'nptable', before='paragraph')