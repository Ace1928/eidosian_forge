import base64
import typing as t
from ...filter.ldap_converters import as_guid, as_sid
from .client import SyncLDAPClient
def cast_object(self, attribute: str, values: t.List[bytes]) -> t.Any:
    info = self.attribute_types.get(attribute.lower(), None)
    caster: t.Callable[[bytes], t.Any]
    if attribute == 'objectSid':
        caster = as_sid
    elif attribute == 'objectGuid':
        caster = as_guid
    elif not info or not info.syntax:
        caster = _as_str
    elif info.syntax == '1.3.6.1.4.1.1466.115.121.1.7':
        caster = _as_bool
    elif info.syntax in ['1.3.6.1.4.1.1466.115.121.1.27', '1.2.840.113556.1.4.906']:
        caster = _as_int
    elif info.syntax in ['1.3.6.1.4.1.1466.115.121.1.40', '1.2.840.113556.1.4.907', 'OctetString']:
        caster = _as_bytes
    else:
        caster = _as_str
    casted_values: t.List = []
    for v in values:
        casted_values.append(caster(v))
    if info and info.single_value:
        return casted_values[0] if casted_values else None
    else:
        return casted_values