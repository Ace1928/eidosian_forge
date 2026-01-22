from unicodedata import ucd_3_2_0 as unicodedata
def in_table_b1(code):
    return ord(code) in b1_set