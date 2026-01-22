from unicodedata import ucd_3_2_0 as unicodedata
def in_table_c3(code):
    return unicodedata.category(code) == 'Co'