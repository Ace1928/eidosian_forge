import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_def_anchor_(self):
    assert self.is_cur_keyword_('DEF_ANCHOR')
    location = self.cur_token_location_
    name = self.expect_string_()
    self.expect_keyword_('ON')
    gid = self.expect_number_()
    self.expect_keyword_('GLYPH')
    glyph_name = self.expect_name_()
    self.expect_keyword_('COMPONENT')
    component = self.expect_number_()
    if glyph_name in self.anchors_:
        anchor = self.anchors_[glyph_name].resolve(name)
        if anchor is not None and anchor.component == component:
            raise VoltLibError('Anchor "%s" already defined, anchor names are case insensitive' % name, location)
    if self.next_token_ == 'LOCKED':
        locked = True
        self.advance_lexer_()
    else:
        locked = False
    self.expect_keyword_('AT')
    pos = self.parse_pos_()
    self.expect_keyword_('END_ANCHOR')
    anchor = ast.AnchorDefinition(name, gid, glyph_name, component, locked, pos, location=location)
    if glyph_name not in self.anchors_:
        self.anchors_[glyph_name] = SymbolTable()
    self.anchors_[glyph_name].define(name, anchor)
    return anchor