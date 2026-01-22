from unicodedata import ucd_3_2_0 as unicodedata
def in_table_c22(code):
    c = ord(code)
    if c < 128:
        return False
    if unicodedata.category(code) == 'Cc':
        return True
    return c in c22_specials