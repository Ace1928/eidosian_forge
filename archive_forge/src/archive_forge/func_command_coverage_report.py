from __future__ import annotations
import os
import typing as t
from ...io import (
from ...util import (
from ...data import (
from ...provisioning import (
from .combine import (
from . import (
def command_coverage_report(args: CoverageReportConfig) -> None:
    """Generate a console coverage report."""
    host_state = prepare_profiles(args)
    output_files = combine_coverage_files(args, host_state)
    for output_file in output_files:
        if args.group_by or args.stub:
            display.info('>>> Coverage Group: %s' % ' '.join(os.path.basename(output_file).split('=')[1:]))
        if output_file.endswith('-powershell'):
            display.info(_generate_powershell_output_report(args, output_file))
        else:
            options = []
            if args.show_missing:
                options.append('--show-missing')
            if args.include:
                options.extend(['--include', args.include])
            if args.omit:
                options.extend(['--omit', args.omit])
            run_coverage(args, host_state, output_file, 'report', options)