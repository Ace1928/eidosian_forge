import collections
from oslo_config import cfg
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
def convert_json_to_yaml(json_str):
    """Convert AWS JSON template format to Heat YAML format.

    :param json_str: a string containing the AWS JSON template format.
    :returns: the equivalent string containing the Heat YAML format.
    """
    tpl = jsonutils.loads(json_str, object_pairs_hook=collections.OrderedDict)

    def top_level_items(tpl):
        yield ('HeatTemplateFormatVersion', '2012-12-12')
        for k, v in tpl.items():
            if k != 'AWSTemplateFormatVersion':
                yield (k, v)
    return yaml.dump(collections.OrderedDict(top_level_items(tpl)), Dumper=yaml_dumper)