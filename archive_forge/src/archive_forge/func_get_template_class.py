import abc
import collections
import copy
import functools
import hashlib
from stevedore import extension
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine import template_files
from heat.objects import raw_template as template_object
def get_template_class(template_data):
    available_versions = _template_classes.keys()
    version = get_version(template_data, available_versions)
    version_type = version[0]
    try:
        return _template_classes[version]
    except KeyError:
        av_list = sorted([v for k, v in available_versions if k == version_type])
        msg_data = {'version': ': '.join(version), 'version_type': version_type, 'available': ', '.join((v for v in av_list))}
        if len(av_list) > 1:
            explanation = _('"%(version)s". "%(version_type)s" should be one of: %(available)s') % msg_data
        else:
            explanation = _('"%(version)s". "%(version_type)s" should be: %(available)s') % msg_data
        raise exception.InvalidTemplateVersion(explanation=explanation)