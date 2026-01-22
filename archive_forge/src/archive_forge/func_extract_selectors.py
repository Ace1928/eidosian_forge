from __future__ import absolute_import, division, print_function
import copy
import json
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
def extract_selectors(instance):
    selectors = []
    if not instance.spec.selector:
        raise CoreException('{0} {1} does not support the log subresource directly, and no Pod selector was found on the object'.format('/'.join(instance.group, instance.apiVersion), instance.kind))
    if not (instance.spec.selector.matchLabels or instance.spec.selector.matchExpressions):
        for k, v in dict(instance.spec.selector).items():
            selectors.append('{0}={1}'.format(k, v))
        return selectors
    if instance.spec.selector.matchLabels:
        for k, v in dict(instance.spec.selector.matchLabels).items():
            selectors.append('{0}={1}'.format(k, v))
    if instance.spec.selector.matchExpressions:
        for expression in instance.spec.selector.matchExpressions:
            operator = expression.operator
            if operator == 'Exists':
                selectors.append(expression.key)
            elif operator == 'DoesNotExist':
                selectors.append('!{0}'.format(expression.key))
            elif operator in ['In', 'NotIn']:
                selectors.append('{key} {operator} {values}'.format(key=expression.key, operator=operator.lower(), values='({0})'.format(', '.join(expression.values))))
            else:
                raise CoreException('The k8s_log module does not support the {0} matchExpression operator'.format(operator.lower()))
    return selectors