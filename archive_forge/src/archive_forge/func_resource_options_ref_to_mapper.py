from keystone.common import validation
from keystone.i18n import _
def resource_options_ref_to_mapper(ref, option_class):
    """Convert the _resource_options property-dict to options attr map.

    The model must have the resource option mapper located in the
    ``_resource_option_mapper`` attribute.

    The model must have the resource option registry located in the
    ``resource_options_registry`` attribute.

    The option dict with key(opt_id), value(opt_value) will be pulled from
    ``ref._resource_options``.

    NOTE: This function MUST be called within the active writer session
          context!

    :param ref: The DB model reference that is actually stored to the
                backend.
    :param option_class: Class that is used to store the resource option
                         in the DB.
    """
    options = getattr(ref, '_resource_options', None)
    if options is not None:
        delattr(ref, '_resource_options')
    else:
        options = {}
    set_options = set(ref._resource_option_mapper.keys())
    clear_options = set_options.difference(ref.resource_options_registry.option_ids)
    options.update({x: None for x in clear_options})
    for r_opt_id, r_opt_value in options.items():
        if r_opt_value is None:
            ref._resource_option_mapper.pop(r_opt_id, None)
        else:
            opt_obj = option_class(option_id=r_opt_id, option_value=r_opt_value)
            ref._resource_option_mapper[r_opt_id] = opt_obj