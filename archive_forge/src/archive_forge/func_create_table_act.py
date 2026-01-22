from pyparsing import Literal, Word, delimitedList \
def create_table_act(toks):
    return '"%(tablename)s" [\n\t label="<%(tablename)s> %(tablename)s | %(columns)s"\n\t shape="record"\n];' % toks