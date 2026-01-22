from unicodedata import ucd_3_2_0 as unicodedata
def map_table_b3(code):
    r = b3_exceptions.get(ord(code))
    if r is not None:
        return r
    return code.lower()