import codecs
import re
import typing as t
from .exceptions import (
def convert_to_utf8(http_headers, data, result):
    """Detect and convert the character encoding to UTF-8.

    http_headers is a dictionary
    data is a raw string (not Unicode)"""
    bom_encoding = ''
    xml_encoding = ''
    if data[:4] == codecs.BOM_UTF32_BE:
        bom_encoding = 'utf-32be'
        data = data[4:]
    elif data[:4] == codecs.BOM_UTF32_LE:
        bom_encoding = 'utf-32le'
        data = data[4:]
    elif data[:2] == codecs.BOM_UTF16_BE and data[2:4] != ZERO_BYTES:
        bom_encoding = 'utf-16be'
        data = data[2:]
    elif data[:2] == codecs.BOM_UTF16_LE and data[2:4] != ZERO_BYTES:
        bom_encoding = 'utf-16le'
        data = data[2:]
    elif data[:3] == codecs.BOM_UTF8:
        bom_encoding = 'utf-8'
        data = data[3:]
    elif data[:4] == EBCDIC_MARKER:
        bom_encoding = 'cp037'
    elif data[:4] == UTF16BE_MARKER:
        bom_encoding = 'utf-16be'
    elif data[:4] == UTF16LE_MARKER:
        bom_encoding = 'utf-16le'
    elif data[:4] == UTF32BE_MARKER:
        bom_encoding = 'utf-32be'
    elif data[:4] == UTF32LE_MARKER:
        bom_encoding = 'utf-32le'
    tempdata = data
    try:
        if bom_encoding:
            tempdata = data.decode(bom_encoding).encode('utf-8')
    except (UnicodeDecodeError, LookupError):
        xml_encoding_match = None
    else:
        xml_encoding_match = RE_XML_PI_ENCODING.match(tempdata)
    if xml_encoding_match:
        xml_encoding = xml_encoding_match.groups()[0].decode('utf-8').lower()
        if bom_encoding and xml_encoding in ('u16', 'utf-16', 'utf16', 'utf_16', 'u32', 'utf-32', 'utf32', 'utf_32', 'iso-10646-ucs-2', 'iso-10646-ucs-4', 'csucs4', 'csunicode', 'ucs-2', 'ucs-4'):
            xml_encoding = bom_encoding
    http_content_type = http_headers.get('content-type') or ''
    http_content_type, http_encoding = parse_content_type(http_content_type)
    acceptable_content_type = 0
    application_content_types = ('application/xml', 'application/xml-dtd', 'application/xml-external-parsed-entity')
    text_content_types = ('text/xml', 'text/xml-external-parsed-entity')
    if http_content_type in application_content_types or (http_content_type.startswith('application/') and http_content_type.endswith('+xml')):
        acceptable_content_type = 1
        rfc3023_encoding = http_encoding or xml_encoding or 'utf-8'
    elif http_content_type in text_content_types or (http_content_type.startswith('text/') and http_content_type.endswith('+xml')):
        acceptable_content_type = 1
        rfc3023_encoding = http_encoding or 'us-ascii'
    elif http_content_type.startswith('text/'):
        rfc3023_encoding = http_encoding or 'us-ascii'
    elif http_headers and 'content-type' not in http_headers:
        rfc3023_encoding = xml_encoding or 'iso-8859-1'
    else:
        rfc3023_encoding = xml_encoding or 'utf-8'
    if rfc3023_encoding.lower() == 'gb2312':
        rfc3023_encoding = 'gb18030'
    if xml_encoding.lower() == 'gb2312':
        xml_encoding = 'gb18030'
    error = None
    if http_headers and (not acceptable_content_type):
        if 'content-type' in http_headers:
            msg = '%s is not an XML media type' % http_headers['content-type']
        else:
            msg = 'no Content-type specified'
        error = NonXMLContentType(msg)
    known_encoding = 0
    tried_encodings = []
    for proposed_encoding in (rfc3023_encoding, xml_encoding, bom_encoding, lazy_chardet_encoding, 'utf-8', 'windows-1252', 'iso-8859-2'):
        if callable(proposed_encoding):
            proposed_encoding = proposed_encoding(data)
        if not proposed_encoding:
            continue
        if proposed_encoding in tried_encodings:
            continue
        tried_encodings.append(proposed_encoding)
        try:
            data = data.decode(proposed_encoding)
        except (UnicodeDecodeError, LookupError):
            pass
        else:
            known_encoding = 1
            new_declaration = "<?xml version='1.0' encoding='utf-8'?>"
            if RE_XML_DECLARATION.search(data):
                data = RE_XML_DECLARATION.sub(new_declaration, data)
            else:
                data = new_declaration + '\n' + data
            data = data.encode('utf-8')
            break
    if not known_encoding:
        error = CharacterEncodingUnknown('document encoding unknown, I tried ' + '%s, %s, utf-8, windows-1252, and iso-8859-2 but nothing worked' % (rfc3023_encoding, xml_encoding))
        rfc3023_encoding = ''
    elif proposed_encoding != rfc3023_encoding:
        error = CharacterEncodingOverride('document declared as %s, but parsed as %s' % (rfc3023_encoding, proposed_encoding))
        rfc3023_encoding = proposed_encoding
    result['encoding'] = rfc3023_encoding
    if error:
        result['bozo'] = True
        result['bozo_exception'] = error
    return data