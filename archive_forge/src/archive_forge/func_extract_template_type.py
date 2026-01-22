from oslo_utils import strutils
from heat.common.i18n import _
def extract_template_type(subject):
    template_type = subject.lower()
    if template_type not in ('cfn', 'hot'):
        raise ValueError(_('Invalid template type "%(value)s", valid types are: cfn, hot.') % {'value': subject})
    return template_type