def parse_accept_header(header_value):
    """Parses the Accept: header.

    The value of the header as a string should be passed in; without
    the header name itself.
    
    This will parse the value of any of the HTTP headers "Accept",
    "Accept-Charset", "Accept-Encoding", or "Accept-Language".  These
    headers are similarly formatted, in that they are a list of items
    with associated quality factors.  The quality factor, or qvalue,
    is a number in the range [0.0..1.0] which indicates the relative
    preference of each item.

    This function returns a list of those items, sorted by preference
    (from most-prefered to least-prefered).  Each item in the returned
    list is actually a tuple consisting of:

       ( item_name, item_parms, qvalue, accept_parms )

    As an example, the following string,
        text/plain; charset="utf-8"; q=.5; columns=80
    would be parsed into this resulting tuple,
        ( 'text/plain', [('charset','utf-8')], 0.5, [('columns','80')] )

    The value of the returned item_name depends upon which header is
    being parsed, but for example it may be a MIME content or media
    type (without parameters), a language tag, or so on.  Any optional
    parameters (delimited by semicolons) occuring before the "q="
    attribute will be in the item_parms list as (attribute,value)
    tuples in the same order as they appear in the header.  Any quoted
    values will have been unquoted and unescaped.

    The qvalue is a floating point number in the inclusive range 0.0
    to 1.0, and roughly indicates the preference for this item.
    Values outside this range will be capped to the closest extreme.

         (!) Note that a qvalue of 0 indicates that the item is
         explicitly NOT acceptable to the user agent, and should be
         handled differently by the caller.

    The accept_parms, like the item_parms, is a list of any attributes
    occuring after the "q=" attribute, and will be in the list as
    (attribute,value) tuples in the same order as they occur.
    Usually accept_parms will be an empty list, as the HTTP spec
    allows these extra parameters in the syntax but does not
    currently define any possible values.

    All empty items will be removed from the list.  However, duplicate
    or conflicting values are not detected or handled in any way by
    this function.
    """

    def parse_mt_only(s, start):
        mt, k = parse_media_type(s, start, with_parameters=False)
        ct = content_type()
        ct.major = mt[0]
        ct.minor = mt[1]
        return (ct, k)
    alist, k = parse_qvalue_accept_list(header_value, item_parser=parse_mt_only)
    if k < len(header_value):
        raise ParseError('Accept header is invalid', header_value, k)
    ctlist = []
    for ct, ctparms, q, acptparms in alist:
        if ctparms:
            ct.set_parameters(dict(ctparms))
        ctlist.append((ct, q, acptparms))
    return ctlist