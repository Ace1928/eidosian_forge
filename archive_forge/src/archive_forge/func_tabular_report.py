from __future__ import annotations
import sys
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage.exceptions import ConfigError, NoDataError
from coverage.misc import human_sorted_items
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
def tabular_report(self) -> None:
    """Writes tabular report formats."""
    header = ['Name', 'Stmts', 'Miss']
    if self.branches:
        header += ['Branch', 'BrPart']
    header += ['Cover']
    if self.config.show_missing:
        header += ['Missing']
    column_order = dict(name=0, stmts=1, miss=2, cover=-1)
    if self.branches:
        column_order.update(dict(branch=3, brpart=4))
    lines_values = []
    for fr, analysis in self.fr_analysis:
        nums = analysis.numbers
        args = [fr.relative_filename(), nums.n_statements, nums.n_missing]
        if self.branches:
            args += [nums.n_branches, nums.n_partial_branches]
        args += [nums.pc_covered_str]
        if self.config.show_missing:
            args += [analysis.missing_formatted(branches=True)]
        args += [nums.pc_covered]
        lines_values.append(args)
    sort_option = (self.config.sort or 'name').lower()
    reverse = False
    if sort_option[0] == '-':
        reverse = True
        sort_option = sort_option[1:]
    elif sort_option[0] == '+':
        sort_option = sort_option[1:]
    sort_idx = column_order.get(sort_option)
    if sort_idx is None:
        raise ConfigError(f'Invalid sorting option: {self.config.sort!r}')
    if sort_option == 'name':
        lines_values = human_sorted_items(lines_values, reverse=reverse)
    else:
        lines_values.sort(key=lambda line: (line[sort_idx], line[0]), reverse=reverse)
    total_line = ['TOTAL', self.total.n_statements, self.total.n_missing]
    if self.branches:
        total_line += [self.total.n_branches, self.total.n_partial_branches]
    total_line += [self.total.pc_covered_str]
    if self.config.show_missing:
        total_line += ['']
    end_lines = []
    if self.config.skip_covered and self.skipped_count:
        file_suffix = 's' if self.skipped_count > 1 else ''
        end_lines.append(f'\n{self.skipped_count} file{file_suffix} skipped due to complete coverage.')
    if self.config.skip_empty and self.empty_count:
        file_suffix = 's' if self.empty_count > 1 else ''
        end_lines.append(f'\n{self.empty_count} empty file{file_suffix} skipped.')
    if self.output_format == 'markdown':
        formatter = self._report_markdown
    else:
        formatter = self._report_text
    formatter(header, lines_values, total_line, end_lines)