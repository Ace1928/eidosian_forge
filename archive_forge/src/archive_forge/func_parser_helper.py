import sys
import six
def parser_helper(key, val):
    """
    Helper for parser function
    @param key:
    @param val:
    """
    start_bracket = key.find('[')
    end_bracket = key.find(']')
    pdict = {}
    if has_variable_name(key):
        pdict[key[:key.find('[')]] = parser_helper(key[start_bracket:], val)
    elif more_than_one_index(key):
        newkey = get_key(key)
        newkey = int(newkey) if is_number(newkey) else newkey
        pdict[newkey] = parser_helper(key[end_bracket + 1:], val)
    else:
        newkey = key
        if start_bracket != -1:
            newkey = get_key(key)
            if newkey is None:
                raise MalformedQueryStringError
        newkey = int(newkey) if is_number(newkey) else newkey
        if key == u'[]':
            val = int(val) if is_number(val) else val
        pdict[newkey] = val
    return pdict