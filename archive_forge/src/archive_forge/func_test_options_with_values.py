import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_options_with_values(self):
    options, sources = parse_command_line(['--embed=huhu', '-I/test/include/dir1', '--include-dir=/test/include/dir2', '--include-dir', '/test/include/dir3', '--working=/work/dir', 'source.pyx', '--output-file=/output/dir', '--pre-import=/pre/import', '--cleanup=3', '--annotate-coverage=cov.xml', '--gdb-outdir=/gdb/outdir', '--directive=wraparound=false'])
    self.assertEqual(sources, ['source.pyx'])
    self.assertEqual(Options.embed, 'huhu')
    self.assertEqual(options.include_path, ['/test/include/dir1', '/test/include/dir2', '/test/include/dir3'])
    self.assertEqual(options.working_path, '/work/dir')
    self.assertEqual(options.output_file, '/output/dir')
    self.assertEqual(Options.pre_import, '/pre/import')
    self.assertEqual(Options.generate_cleanup_code, 3)
    self.assertTrue(Options.annotate)
    self.assertEqual(Options.annotate_coverage_xml, 'cov.xml')
    self.assertTrue(options.gdb_debug)
    self.assertEqual(options.output_dir, '/gdb/outdir')
    self.assertEqual(options.compiler_directives['wraparound'], False)