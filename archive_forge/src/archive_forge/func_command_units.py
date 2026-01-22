from __future__ import annotations
import os
import sys
import typing as t
from ...constants import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...config import (
from ...coverage_util import (
from ...data import (
from ...executor import (
from ...python_requirements import (
from ...content_config import (
from ...host_configs import (
from ...provisioning import (
from ...pypi_proxy import (
from ...host_profiles import (
def command_units(args: UnitsConfig) -> None:
    """Run unit tests."""
    handle_layout_messages(data_context().content.unit_messages)
    changes = get_changes_filter(args)
    require = args.require + changes
    include = walk_internal_targets(walk_units_targets(), args.include, args.exclude, require)
    paths = [target.path for target in include]
    content_config = get_content_config(args)
    supported_remote_python_versions = content_config.modules.python_versions
    if content_config.modules.controller_only:
        module_paths = []
        module_utils_paths = []
    else:
        module_paths = [path for path in paths if is_subdir(path, data_context().content.unit_module_path)]
        module_utils_paths = [path for path in paths if is_subdir(path, data_context().content.unit_module_utils_path)]
    controller_paths = sorted((path for path in set(paths) - set(module_paths) - set(module_utils_paths)))
    remote_paths = module_paths or module_utils_paths
    test_context_paths = {TestContext.modules: module_paths, TestContext.module_utils: module_utils_paths, TestContext.controller: controller_paths}
    if not paths:
        raise AllTargetsSkipped()
    targets = t.cast(list[PosixConfig], args.targets)
    target_versions: dict[str, PosixConfig] = {target.python.version: target for target in targets}
    skipped_versions = args.host_settings.skipped_python_versions
    warn_versions = []
    test_versions = [version for version in target_versions if version in REMOTE_ONLY_PYTHON_VERSIONS and version not in supported_remote_python_versions]
    if test_versions:
        for version in test_versions:
            display.warning(f'Skipping unit tests on Python {version} because it is not supported by this collection. Supported Python versions are: {', '.join(content_config.python_versions)}')
        warn_versions.extend(test_versions)
        if warn_versions == list(target_versions):
            raise AllTargetsSkipped()
    if not remote_paths:
        test_versions = [version for version in target_versions if version in REMOTE_ONLY_PYTHON_VERSIONS and version not in warn_versions]
        if test_versions:
            for version in test_versions:
                display.warning(f'Skipping unit tests on Python {version} because it is only supported by module/module_utils unit tests. No module/module_utils unit tests were selected.')
            warn_versions.extend(test_versions)
            if warn_versions == list(target_versions):
                raise AllTargetsSkipped()
    if not controller_paths:
        test_versions = [version for version in target_versions if version not in supported_remote_python_versions and version not in warn_versions]
        if test_versions:
            for version in test_versions:
                display.warning(f'Skipping unit tests on Python {version} because it is not supported by module/module_utils unit tests of this collection. Supported Python versions are: {', '.join(supported_remote_python_versions)}')
            warn_versions.extend(test_versions)
            if warn_versions == list(target_versions):
                raise AllTargetsSkipped()
    host_state = prepare_profiles(args, targets_use_pypi=True)
    if args.delegate:
        raise Delegate(host_state=host_state, require=changes, exclude=args.exclude)
    test_sets = []
    if args.requirements_mode != 'skip':
        configure_pypi_proxy(args, host_state.controller_profile)
    for version in SUPPORTED_PYTHON_VERSIONS:
        if version not in target_versions and version not in skipped_versions:
            continue
        test_candidates = []
        for test_context, paths in test_context_paths.items():
            if test_context == TestContext.controller:
                if version not in CONTROLLER_PYTHON_VERSIONS:
                    continue
            elif version not in supported_remote_python_versions:
                continue
            if not paths:
                continue
            env = ansible_environment(args)
            env.update(PYTHONPATH=get_units_ansible_python_path(args, test_context), ANSIBLE_CONTROLLER_MIN_PYTHON_VERSION=CONTROLLER_MIN_PYTHON_VERSION)
            test_candidates.append((test_context, paths, env))
        if not test_candidates:
            continue
        if version in skipped_versions:
            display.warning('Skipping unit tests on Python %s because it could not be found.' % version)
            continue
        target_profiles: dict[str, PosixProfile] = {profile.config.python.version: profile for profile in host_state.targets(PosixProfile)}
        target_profile = target_profiles[version]
        final_candidates = [(test_context, target_profile.python, paths, env) for test_context, paths, env in test_candidates]
        controller = any((test_context == TestContext.controller for test_context, python, paths, env in final_candidates))
        if args.requirements_mode != 'skip':
            install_requirements(args, target_profile.python, ansible=controller, command=True, controller=False)
        test_sets.extend(final_candidates)
    if args.requirements_mode == 'only':
        sys.exit()
    for test_context, python, paths, env in test_sets:
        if str_to_version(python.version) < (3, 8):
            config_name = 'legacy.ini'
        else:
            config_name = 'default.ini'
        cmd = ['pytest', '-r', 'a', '-n', str(args.num_workers) if args.num_workers else 'auto', '--color', 'yes' if args.color else 'no', '-p', 'no:cacheprovider', '-c', os.path.join(ANSIBLE_TEST_DATA_ROOT, 'pytest', 'config', config_name), '--junit-xml', os.path.join(ResultType.JUNIT.path, 'python%s-%s-units.xml' % (python.version, test_context)), '--strict-markers', '--rootdir', data_context().content.root, '--confcutdir', data_context().content.root]
        if not data_context().content.collection:
            cmd.append('--durations=25')
        plugins = []
        if args.coverage:
            plugins.append('ansible_pytest_coverage')
        if data_context().content.collection:
            plugins.append('ansible_pytest_collections')
        plugins.append('ansible_forked')
        if plugins:
            env['PYTHONPATH'] += ':%s' % os.path.join(ANSIBLE_TEST_TARGET_ROOT, 'pytest/plugins')
            env['PYTEST_PLUGINS'] = ','.join(plugins)
        if args.collect_only:
            cmd.append('--collect-only')
        if args.verbosity:
            cmd.append('-' + 'v' * args.verbosity)
        cmd.extend(paths)
        display.info('Unit test %s with Python %s' % (test_context, python.version))
        try:
            cover_python(args, python, cmd, test_context, env, capture=False)
        except SubprocessError as ex:
            if ex.status != 5:
                raise