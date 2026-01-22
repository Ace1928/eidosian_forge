from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
def construct_yaml_seq(self, node):
    ret_val = list(super(_LvObjectConstructor, self).construct_yaml_seq(node))[0]
    ret_val.value = ret_val
    return ret_val