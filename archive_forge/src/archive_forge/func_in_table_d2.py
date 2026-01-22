from unicodedata import ucd_3_2_0 as unicodedata
def in_table_d2(code):
    return unicodedata.bidirectional(code) == 'L'