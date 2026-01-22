from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def create_ascii_table(headers: list, rows: list, title: str):
    """Creates a pretty table from a list of rows, minimal version of `tabulate`."""
    sep_char, in_between = ('│', '─')
    column_widths = []
    for i in range(len(headers)):
        column_values = [row[i] for row in rows] + [headers[i]]
        max_column_width = max((len(value) for value in column_values))
        column_widths.append(max_column_width)
    formats = [f'%{column_widths[i]}s' for i in range(len(rows[0]))]
    pattern = f'{sep_char}{sep_char.join(formats)}{sep_char}'
    diff = 0

    def make_row(left_char, middle_char, right_char):
        return f'{left_char}{middle_char.join([in_between * n for n in column_widths])}{in_between * diff}{right_char}'
    separator = make_row('├', '┼', '┤')
    if len(title) > sum(column_widths):
        diff = abs(len(title) - len(separator))
        column_widths[-1] += diff
    separator = make_row('├', '┼', '┤')
    initial_rows = [make_row('┌', in_between, '┐'), f'{sep_char}{title.center(len(separator) - 2)}{sep_char}', make_row('├', '┬', '┤')]
    table = '\n'.join(initial_rows) + '\n'
    column_widths[-1] += diff
    centered_line = [text.center(column_widths[i]) for i, text in enumerate(headers)]
    table += f'{pattern % tuple(centered_line)}\n{separator}\n'
    for i, line in enumerate(rows):
        centered_line = [t.center(column_widths[i]) for i, t in enumerate(line)]
        table += f'{pattern % tuple(centered_line)}\n'
    table += f'└{'┴'.join([in_between * n for n in column_widths])}┘'
    return table