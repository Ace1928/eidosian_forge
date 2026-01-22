import collections
from oslo_config import cfg
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
def top_level_items(tpl):
    yield ('HeatTemplateFormatVersion', '2012-12-12')
    for k, v in tpl.items():
        if k != 'AWSTemplateFormatVersion':
            yield (k, v)