import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def write_dl(self, rows: t.Sequence[t.Tuple[str, str]], col_max: int=30, col_spacing: int=2) -> None:
    """Writes a definition list into the buffer.  This is how options
        and commands are usually formatted.

        :param rows: a list of two item tuples for the terms and values.
        :param col_max: the maximum width of the first column.
        :param col_spacing: the number of spaces between the first and
                            second column.
        """
    rows = list(rows)
    widths = measure_table(rows)
    if len(widths) != 2:
        raise TypeError('Expected two columns for definition list')
    first_col = min(widths[0], col_max) + col_spacing
    for first, second in iter_rows(rows, len(widths)):
        self.write(f'{'':>{self.current_indent}}{first}')
        if not second:
            self.write('\n')
            continue
        if term_len(first) <= first_col - col_spacing:
            self.write(' ' * (first_col - term_len(first)))
        else:
            self.write('\n')
            self.write(' ' * (first_col + self.current_indent))
        text_width = max(self.width - first_col - 2, 10)
        wrapped_text = wrap_text(second, text_width, preserve_paragraphs=True)
        lines = wrapped_text.splitlines()
        if lines:
            self.write(f'{lines[0]}\n')
            for line in lines[1:]:
                self.write(f'{'':>{first_col + self.current_indent}}{line}\n')
        else:
            self.write('\n')