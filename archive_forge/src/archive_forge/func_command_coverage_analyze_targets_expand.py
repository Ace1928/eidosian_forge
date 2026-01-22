from __future__ import annotations
import typing as t
from .....io import (
from .....executor import (
from .....provisioning import (
from . import (
def command_coverage_analyze_targets_expand(args: CoverageAnalyzeTargetsExpandConfig) -> None:
    """Expand target names in an aggregated coverage file."""
    host_state = prepare_profiles(args)
    if args.delegate:
        raise Delegate(host_state=host_state)
    covered_targets, covered_path_arcs, covered_path_lines = read_report(args.input_file)
    report = dict(arcs=expand_indexes(covered_path_arcs, covered_targets, format_arc), lines=expand_indexes(covered_path_lines, covered_targets, format_line))
    if not args.explain:
        write_json_file(args.output_file, report, encoder=SortedSetEncoder)