import argparse
import contextlib
import os
import sys
from configparser import ConfigParser
from typing import Dict, List, Union
from docformatter import __pkginfo__
def do_parse_arguments(self) -> None:
    """Parse configuration file and command line arguments."""
    changes = self.parser.add_mutually_exclusive_group()
    changes.add_argument('-i', '--in-place', action='store_true', default=self.flargs_dct.get('in-place', 'false').lower() == 'true', help='make changes to files instead of printing diffs')
    changes.add_argument('-c', '--check', action='store_true', default=self.flargs_dct.get('check', 'false').lower() == 'true', help='only check and report incorrectly formatted files')
    self.parser.add_argument('-d', '--diff', action='store_true', default=self.flargs_dct.get('diff', 'false').lower() == 'true', help='when used with `--check` or `--in-place`, also what changes would be made')
    self.parser.add_argument('-r', '--recursive', action='store_true', default=self.flargs_dct.get('recursive', 'false').lower() == 'true', help='drill down directories recursively')
    self.parser.add_argument('-e', '--exclude', nargs='*', default=self.flargs_dct.get('exclude', None), help='in recursive mode, exclude directories and files by names')
    self.parser.add_argument('-n', '--non-cap', action='store', nargs='*', default=self.flargs_dct.get('non-cap', None), help='list of words not to capitalize when they appear as the first word in the summary')
    self.parser.add_argument('--black', action='store_true', default=self.flargs_dct.get('black', 'false').lower() == 'true', help='make formatting compatible with standard black options (default: False)')
    self.args = self.parser.parse_known_args(self.args_lst[1:])[0]
    if self.args.black:
        _default_wrap_summaries = 88
        _default_wrap_descriptions = 88
        _default_pre_summary_space = 'true'
    else:
        _default_wrap_summaries = 79
        _default_wrap_descriptions = 72
        _default_pre_summary_space = 'false'
    self.parser.add_argument('-s', '--style', default=self.flargs_dct.get('style', 'sphinx'), help='name of the docstring style to use when formatting parameter lists (default: sphinx)')
    self.parser.add_argument('--rest-section-adorns', type=str, dest='rest_section_adorns', default=self.flargs_dct.get('rest_section_adorns', '[!\\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]{4,}'), help='regex for identifying reST section header adornments')
    self.parser.add_argument('--wrap-summaries', default=int(self.flargs_dct.get('wrap-summaries', _default_wrap_summaries)), type=int, metavar='length', help='wrap long summary lines at this length; set to 0 to disable wrapping (default: 79, 88 with --black option)')
    self.parser.add_argument('--wrap-descriptions', default=int(self.flargs_dct.get('wrap-descriptions', _default_wrap_descriptions)), type=int, metavar='length', help='wrap descriptions at this length; set to 0 to disable wrapping (default: 72, 88 with --black option)')
    self.parser.add_argument('--force-wrap', action='store_true', default=self.flargs_dct.get('force-wrap', 'false').lower() == 'true', help='force descriptions to be wrapped even if it may result in a mess (default: False)')
    self.parser.add_argument('--tab-width', type=int, dest='tab_width', metavar='width', default=int(self.flargs_dct.get('tab-width', 1)), help='tabs in indentation are this many characters when wrapping lines (default: 1)')
    self.parser.add_argument('--blank', dest='post_description_blank', action='store_true', default=self.flargs_dct.get('blank', 'false').lower() == 'true', help='add blank line after description (default: False)')
    self.parser.add_argument('--pre-summary-newline', action='store_true', default=self.flargs_dct.get('pre-summary-newline', 'false').lower() == 'true', help='add a newline before the summary of a multi-line docstring (default: False)')
    self.parser.add_argument('--pre-summary-space', action='store_true', default=self.flargs_dct.get('pre-summary-space', _default_pre_summary_space).lower() == 'true', help='add a space after the opening triple quotes (default: False)')
    self.parser.add_argument('--make-summary-multi-line', action='store_true', default=self.flargs_dct.get('make-summary-multi-line', 'false').lower() == 'true', help='add a newline before and after the summary of a one-line docstring (default: False)')
    self.parser.add_argument('--close-quotes-on-newline', action='store_true', default=self.flargs_dct.get('close-quotes-on-newline', 'false').lower() == 'true', help='place closing triple quotes on a new-line when a one-line docstring wraps to two or more lines (default: False)')
    self.parser.add_argument('--range', metavar='line', dest='line_range', default=self.flargs_dct.get('range', None), type=int, nargs=2, help='apply docformatter to docstrings between these lines; line numbers are indexed at 1 (default: None)')
    self.parser.add_argument('--docstring-length', metavar='length', dest='length_range', default=self.flargs_dct.get('docstring-length', None), type=int, nargs=2, help='apply docformatter to docstrings of given length range (default: None)')
    self.parser.add_argument('--non-strict', action='store_true', default=self.flargs_dct.get('non-strict', 'false').lower() == 'true', help="don't strictly follow reST syntax to identify lists (see issue #67) (default: False)")
    self.parser.add_argument('--config', default=self.config_file, help='path to file containing docformatter options')
    self.parser.add_argument('--version', action='version', version=f'%(prog)s {__pkginfo__.__version__}')
    self.parser.add_argument('files', nargs='+', help="files to format or '-' for standard in")
    self.args = self.parser.parse_args(self.args_lst[1:])
    if self.args.line_range:
        if self.args.line_range[0] <= 0:
            self.parser.error('--range must be positive numbers')
        if self.args.line_range[0] > self.args.line_range[1]:
            self.parser.error('First value of --range should be less than or equal to the second')
    if self.args.length_range:
        if self.args.length_range[0] <= 0:
            self.parser.error('--docstring-length must be positive numbers')
        if self.args.length_range[0] > self.args.length_range[1]:
            self.parser.error('First value of --docstring-length should be less than or equal to the second')