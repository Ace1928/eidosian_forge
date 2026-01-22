from __future__ import annotations
import collections.abc as c
import re
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def command_coverage_analyze_targets_filter(args: CoverageAnalyzeTargetsFilterConfig) -> None:
    """Filter target names in an aggregated coverage file."""
    host_state = prepare_profiles(args)
    if args.delegate:
        raise Delegate(host_state=host_state)
    covered_targets, covered_path_arcs, covered_path_lines = read_report(args.input_file)

    def pass_target_key(value: TargetKey) -> TargetKey:
        """Return the given target key unmodified."""
        return value
    filtered_path_arcs = expand_indexes(covered_path_arcs, covered_targets, pass_target_key)
    filtered_path_lines = expand_indexes(covered_path_lines, covered_targets, pass_target_key)
    include_targets = set(args.include_targets) if args.include_targets else None
    exclude_targets = set(args.exclude_targets) if args.exclude_targets else None
    include_path = re.compile(args.include_path) if args.include_path else None
    exclude_path = re.compile(args.exclude_path) if args.exclude_path else None

    def path_filter_func(path: str) -> bool:
        """Return True if the given path should be included, otherwise return False."""
        if include_path and (not re.search(include_path, path)):
            return False
        if exclude_path and re.search(exclude_path, path):
            return False
        return True

    def target_filter_func(targets: set[str]) -> set[str]:
        """Filter the given targets and return the result based on the defined includes and excludes."""
        if include_targets:
            targets &= include_targets
        if exclude_targets:
            targets -= exclude_targets
        return targets
    filtered_path_arcs = filter_data(filtered_path_arcs, path_filter_func, target_filter_func)
    filtered_path_lines = filter_data(filtered_path_lines, path_filter_func, target_filter_func)
    target_indexes: TargetIndexes = {}
    indexed_path_arcs = generate_indexes(target_indexes, filtered_path_arcs)
    indexed_path_lines = generate_indexes(target_indexes, filtered_path_lines)
    report = make_report(target_indexes, indexed_path_arcs, indexed_path_lines)
    write_report(args, report, args.output_file)