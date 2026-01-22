import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def parse_kerberos_token(token: typing.Union['KerberosV5Msg', 'PAData', 'PAETypeInfo2', 'EncryptedData', 'Ticket', 'KdcReqBody'], secret: typing.Optional[str]=None, encoding: typing.Optional[str]=None) -> typing.Union[str, typing.Dict[str, typing.Any]]:
    """Parses a KerberosV5Msg object to a dict."""
    text_encoding = encoding if encoding else 'utf-8'

    def parse_default(value: typing.Any) -> typing.Any:
        return value

    def parse_datetime(value: datetime.datetime) -> str:
        return value.isoformat()

    def parse_text(value: bytes) -> str:
        return to_text(value, encoding=text_encoding, errors='replace')

    def parse_bytes(value: bytes) -> str:
        return base64.b16encode(value).decode()

    def parse_principal_name(value: PrincipalName) -> typing.Dict[str, typing.Any]:
        return {'name-type': parse_enum(value.name_type), 'name-string': [parse_text(v) for v in value.value]}

    def parse_host_address(value: HostAddress) -> typing.Dict[str, typing.Any]:
        return {'addr-type': parse_enum(value.addr_type), 'address': parse_text(value.value)}

    def parse_token(value: typing.Any) -> typing.Union[str, typing.Dict[str, typing.Any]]:
        return parse_kerberos_token(value, secret, text_encoding)
    if isinstance(token, bytes):
        return parse_bytes(token)
    msg = {}
    for name, attr_name, attr_type in getattr(token, 'PARSE_MAP', {}):
        attr_value = getattr(token, attr_name)
        parse_args = []
        if isinstance(attr_type, tuple):
            parse_args.append(attr_type[1])
            attr_type = attr_type[0]
        parse_func: typing.Callable = {ParseType.default: parse_default, ParseType.enum: parse_enum, ParseType.flags: parse_flags, ParseType.datetime: parse_datetime, ParseType.text: parse_text, ParseType.bytes: parse_bytes, ParseType.principal_name: parse_principal_name, ParseType.host_address: parse_host_address, ParseType.token: parse_token}[attr_type]
        if attr_value is None:
            parsed_value = None
        elif isinstance(attr_value, list):
            parsed_value = [parse_func(v, *parse_args) if v is not None else None for v in attr_value]
        else:
            parsed_value = parse_func(attr_value, *parse_args)
        msg[name] = parsed_value
    return msg