from __future__ import annotations
import os
import typing as t
from .....encoding import (
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def command_coverage_analyze_targets_missing(args: CoverageAnalyzeTargetsMissingConfig) -> None:
    """Identify aggregated coverage in one file missing from another."""
    host_state = prepare_profiles(args)
    if args.delegate:
        raise Delegate(host_state=host_state)
    from_targets, from_path_arcs, from_path_lines = read_report(args.from_file)
    to_targets, to_path_arcs, to_path_lines = read_report(args.to_file)
    target_indexes: TargetIndexes = {}
    if args.only_gaps:
        arcs = find_gaps(from_path_arcs, from_targets, to_path_arcs, target_indexes, args.only_exists)
        lines = find_gaps(from_path_lines, from_targets, to_path_lines, target_indexes, args.only_exists)
    else:
        arcs = find_missing(from_path_arcs, from_targets, to_path_arcs, to_targets, target_indexes, args.only_exists)
        lines = find_missing(from_path_lines, from_targets, to_path_lines, to_targets, target_indexes, args.only_exists)
    report = make_report(target_indexes, arcs, lines)
    write_report(args, report, args.output_file)