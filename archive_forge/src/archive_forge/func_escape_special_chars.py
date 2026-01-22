from pygments.formatter import Formatter
def escape_special_chars(text, table=_escape_table):
    """Escape & and < for Pango Markup."""
    return text.translate(table)