import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_attach_cursive_(self):
    assert self.is_cur_keyword_('ATTACH_CURSIVE')
    location = self.cur_token_location_
    coverages_exit = []
    coverages_enter = []
    while self.next_token_ != 'ENTER':
        self.expect_keyword_('EXIT')
        coverages_exit.append(self.parse_coverage_())
    while self.next_token_ != 'END_ATTACH':
        self.expect_keyword_('ENTER')
        coverages_enter.append(self.parse_coverage_())
    self.expect_keyword_('END_ATTACH')
    position = ast.PositionAttachCursiveDefinition(coverages_exit, coverages_enter, location=location)
    return position