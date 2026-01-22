from keystone.common import validation
from keystone.i18n import _
def ref_mapper_to_dict_options(ref):
    """Convert the values in _resource_option_mapper to options dict.

    NOTE: this is to be called from the relevant `to_dict` methods or
          similar and must be called from within the active session context.

    :param ref: the DB model ref to extract options from
    :returns: Dict of options as expected to be returned out of to_dict in
              the `options` key.
    """
    options = {}
    for opt in ref._resource_option_mapper.values():
        if opt.option_id in ref.resource_options_registry.option_ids:
            r_opt = ref.resource_options_registry.get_option_by_id(opt.option_id)
            if r_opt is not None:
                options[r_opt.option_name] = opt.option_value
    return options