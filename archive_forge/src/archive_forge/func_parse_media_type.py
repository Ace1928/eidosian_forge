def parse_media_type(media_type, start=0, with_parameters=True):
    """Parses a media type (MIME type) designator into it's parts.

    Given a media type string, returns a nested tuple of it's parts.

        ((major,minor,parmlist), chars_consumed)

    where parmlist is a list of tuples of (parm_name, parm_value).
    Quoted-values are appropriately unquoted and unescaped.
    
    If 'with_parameters' is False, then parsing will stop immediately
    after the minor media type; and will not proceed to parse any
    of the semicolon-separated paramters.

    Examples:
        image/png -> (('image','png',[]), 9)
        text/plain; charset="utf-16be"
                  -> (('text','plain',[('charset,'utf-16be')]), 30)

    """
    s = media_type
    pos = start
    ctmaj, k = parse_token(s, pos)
    if k == 0:
        raise ParseError('Media type must be of the form "major/minor".', s, pos)
    pos += k
    if pos >= len(s) or s[pos] != '/':
        raise ParseError('Media type must be of the form "major/minor".', s, pos)
    pos += 1
    ctmin, k = parse_token(s, pos)
    if k == 0:
        raise ParseError('Media type must be of the form "major/minor".', s, pos)
    pos += k
    if with_parameters:
        parmlist, k = parse_parameter_list(s, pos)
        pos += k
    else:
        parmlist = []
    return ((ctmaj, ctmin, parmlist), pos - start)