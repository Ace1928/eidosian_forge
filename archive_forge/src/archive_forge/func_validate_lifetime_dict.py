from neutronclient._i18n import _
from neutronclient.common import exceptions
def validate_lifetime_dict(lifetime_dict):
    for key, value in lifetime_dict.items():
        if key not in lifetime_keys:
            message = _("Lifetime Dictionary KeyError: Reason-Invalid unit key : '%(key)s' not in %(supported_key)s") % {'key': key, 'supported_key': lifetime_keys}
            raise exceptions.CommandError(message)
        if key == 'units' and value not in lifetime_units:
            message = _("Lifetime Dictionary ValueError: Reason-Invalid units : '%(key_value)s' not in %(supported_units)s") % {'key_value': key, 'supported_units': lifetime_units}
            raise exceptions.CommandError(message)
        if key == 'value':
            try:
                if int(value) < 60:
                    raise ValueError()
            except ValueError:
                message = _("Lifetime Dictionary ValueError: Reason-Invalid value should be at least 60:'%(key_value)s' = %(value)s") % {'key_value': key, 'value': value}
                raise exceptions.CommandError(message)
            else:
                lifetime_dict['value'] = int(value)
    return