import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_def_group_(self):
    assert self.is_cur_keyword_('DEF_GROUP')
    location = self.cur_token_location_
    name = self.expect_string_()
    enum = None
    if self.next_token_ == 'ENUM':
        enum = self.parse_enum_()
    self.expect_keyword_('END_GROUP')
    if self.groups_.resolve(name) is not None:
        raise VoltLibError('Glyph group "%s" already defined, group names are case insensitive' % name, location)
    def_group = ast.GroupDefinition(name, enum, location=location)
    self.groups_.define(name, def_group)
    return def_group