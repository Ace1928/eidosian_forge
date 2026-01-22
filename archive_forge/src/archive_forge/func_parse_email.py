import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import Dict, List, Optional, Tuple, Union, cast
def parse_email(data: Union[bytes, str]) -> Tuple[RawMetadata, Dict[str, List[str]]]:
    """Parse a distribution's metadata.

    This function returns a two-item tuple of dicts. The first dict is of
    recognized fields from the core metadata specification. Fields that can be
    parsed and translated into Python's built-in types are converted
    appropriately. All other fields are left as-is. Fields that are allowed to
    appear multiple times are stored as lists.

    The second dict contains all other fields from the metadata. This includes
    any unrecognized fields. It also includes any fields which are expected to
    be parsed into a built-in type but were not formatted appropriately. Finally,
    any fields that are expected to appear only once but are repeated are
    included in this dict.

    """
    raw: Dict[str, Union[str, List[str], Dict[str, str]]] = {}
    unparsed: Dict[str, List[str]] = {}
    if isinstance(data, str):
        parsed = email.parser.Parser(policy=email.policy.compat32).parsestr(data)
    else:
        parsed = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(data)
    for name in frozenset(parsed.keys()):
        name = name.lower()
        headers = parsed.get_all(name)
        value = []
        valid_encoding = True
        for h in headers:
            assert isinstance(h, (email.header.Header, str))
            if isinstance(h, email.header.Header):
                chunks: List[Tuple[bytes, Optional[str]]] = []
                for bin, encoding in email.header.decode_header(h):
                    try:
                        bin.decode('utf8', 'strict')
                    except UnicodeDecodeError:
                        encoding = 'latin1'
                        valid_encoding = False
                    else:
                        encoding = 'utf8'
                    chunks.append((bin, encoding))
                value.append(str(email.header.make_header(chunks)))
            else:
                value.append(h)
        if not valid_encoding:
            unparsed[name] = value
            continue
        raw_name = _EMAIL_TO_RAW_MAPPING.get(name)
        if raw_name is None:
            unparsed[name] = value
            continue
        if raw_name in _STRING_FIELDS and len(value) == 1:
            raw[raw_name] = value[0]
        elif raw_name in _LIST_STRING_FIELDS:
            raw[raw_name] = value
        elif raw_name == 'keywords' and len(value) == 1:
            raw[raw_name] = _parse_keywords(value[0])
        elif raw_name == 'project_urls':
            try:
                raw[raw_name] = _parse_project_urls(value)
            except KeyError:
                unparsed[name] = value
        else:
            unparsed[name] = value
    try:
        payload = _get_payload(parsed, data)
    except ValueError:
        unparsed.setdefault('description', []).append(parsed.get_payload(decode=isinstance(data, bytes)))
    else:
        if payload:
            if 'description' in raw:
                description_header = cast(str, raw.pop('description'))
                unparsed.setdefault('description', []).extend([description_header, payload])
            elif 'description' in unparsed:
                unparsed['description'].append(payload)
            else:
                raw['description'] = payload
    return (cast(RawMetadata, raw), unparsed)