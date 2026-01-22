from unicodedata import ucd_3_2_0 as unicodedata
def in_table_c5(code):
    return unicodedata.category(code) == 'Cs'