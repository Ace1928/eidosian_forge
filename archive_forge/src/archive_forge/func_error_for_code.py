from struct import pack, unpack
def error_for_code(code, text, method, default):
    try:
        return ERROR_MAP[code](text, method, reply_code=code)
    except KeyError:
        return default(text, method, reply_code=code)