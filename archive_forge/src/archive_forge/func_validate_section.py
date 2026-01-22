import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
def validate_section(self, section, sub_section, data, allowed_keys):
    obj_name = section[:-1]
    err_msg = _('"%%s" is not a valid keyword inside a %s definition') % obj_name
    args = {'object_name': obj_name, 'sub_section': sub_section}
    message = _('Each %(object_name)s must contain a %(sub_section)s key.') % args
    for name, attrs in sorted(data.items()):
        if not attrs:
            raise exception.StackValidationFailed(message=message)
        try:
            for attr, attr_value in attrs.items():
                if attr not in allowed_keys:
                    raise KeyError(err_msg % attr)
            if sub_section not in attrs:
                raise exception.StackValidationFailed(message=message)
        except AttributeError:
            message = _('"%(section)s" must contain a map of %(obj_name)s maps. Found a [%(_type)s] instead') % {'section': section, '_type': type(attrs), 'obj_name': obj_name}
            raise exception.StackValidationFailed(message=message)
        except KeyError as e:
            raise exception.StackValidationFailed(message=e.args[0])