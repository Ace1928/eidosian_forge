from __future__ import annotations
import dataclasses
import json
import textwrap
import os
import re
import typing as t
from . import (
from ...test import (
from ...config import (
from ...target import (
from ..integration.cloud import (
from ...io import (
from ...util import (
from ...util_common import (
from ...host_configs import (
def check_posix_targets(self, args: SanityConfig) -> list[SanityMessage]:
    """Check POSIX integration test targets and return messages with any issues found."""
    posix_targets = tuple(walk_posix_integration_targets())
    clouds = get_cloud_platforms(args, posix_targets)
    cloud_targets = ['cloud/%s/' % cloud for cloud in clouds]
    all_cloud_targets = tuple(filter_targets(posix_targets, ['cloud/'], errors=False))
    invalid_cloud_targets = tuple(filter_targets(all_cloud_targets, cloud_targets, include=False, errors=False))
    messages = []
    for target in invalid_cloud_targets:
        for alias in target.aliases:
            if alias.startswith('cloud/') and alias != 'cloud/':
                if any((alias.startswith(cloud_target) for cloud_target in cloud_targets)):
                    continue
                messages.append(SanityMessage('invalid alias `%s`' % alias, '%s/aliases' % target.path))
    messages += self.check_ci_group(targets=tuple(filter_targets(posix_targets, ['cloud/', '%s/generic/' % self.TEST_ALIAS_PREFIX], include=False, errors=False)), find=self.format_test_group_alias('linux').replace('linux', 'posix'), find_incidental=['%s/posix/incidental/' % self.TEST_ALIAS_PREFIX])
    messages += self.check_ci_group(targets=tuple(filter_targets(posix_targets, ['%s/generic/' % self.TEST_ALIAS_PREFIX], errors=False)), find=self.format_test_group_alias('generic'))
    for cloud in clouds:
        if cloud == 'httptester':
            find = self.format_test_group_alias('linux').replace('linux', 'posix')
            find_incidental = ['%s/posix/incidental/' % self.TEST_ALIAS_PREFIX]
        else:
            find = self.format_test_group_alias(cloud, 'generic')
            find_incidental = ['%s/%s/incidental/' % (self.TEST_ALIAS_PREFIX, cloud), '%s/cloud/incidental/' % self.TEST_ALIAS_PREFIX]
        messages += self.check_ci_group(targets=tuple(filter_targets(posix_targets, ['cloud/%s/' % cloud], errors=False)), find=find, find_incidental=find_incidental)
    target_type_groups = {IntegrationTargetType.TARGET: (1, 2), IntegrationTargetType.CONTROLLER: (3, 4, 5), IntegrationTargetType.CONFLICT: (), IntegrationTargetType.UNKNOWN: ()}
    for target in posix_targets:
        if target.name == 'ansible-test-container':
            continue
        if f'{self.TEST_ALIAS_PREFIX}/posix/' not in target.aliases:
            continue
        found_groups = [alias for alias in target.aliases if re.search(f'^{self.TEST_ALIAS_PREFIX}/posix/group[0-9]+/$', alias)]
        expected_groups = [f'{self.TEST_ALIAS_PREFIX}/posix/group{group}/' for group in target_type_groups[target.target_type]]
        valid_groups = [group for group in found_groups if group in expected_groups]
        invalid_groups = [group for group in found_groups if not any((group.startswith(expected_group) for expected_group in expected_groups))]
        if not valid_groups:
            messages.append(SanityMessage(f'Target of type {target.target_type.name} must be in at least one of these groups: {', '.join(expected_groups)}', f'{target.path}/aliases'))
        if invalid_groups:
            messages.append(SanityMessage(f'Target of type {target.target_type.name} cannot be in these groups: {', '.join(invalid_groups)}', f'{target.path}/aliases'))
    return messages