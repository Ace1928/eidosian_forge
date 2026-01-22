import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_pos_(self):
    self.advance_lexer_()
    location = self.cur_token_location_
    assert self.is_cur_keyword_('POS'), location
    adv = None
    dx = None
    dy = None
    adv_adjust_by = {}
    dx_adjust_by = {}
    dy_adjust_by = {}
    if self.next_token_ == 'ADV':
        self.advance_lexer_()
        adv = self.expect_number_()
        while self.next_token_ == 'ADJUST_BY':
            adjustment, size = self.parse_adjust_by_()
            adv_adjust_by[size] = adjustment
    if self.next_token_ == 'DX':
        self.advance_lexer_()
        dx = self.expect_number_()
        while self.next_token_ == 'ADJUST_BY':
            adjustment, size = self.parse_adjust_by_()
            dx_adjust_by[size] = adjustment
    if self.next_token_ == 'DY':
        self.advance_lexer_()
        dy = self.expect_number_()
        while self.next_token_ == 'ADJUST_BY':
            adjustment, size = self.parse_adjust_by_()
            dy_adjust_by[size] = adjustment
    self.expect_keyword_('END_POS')
    return ast.Pos(adv, dx, dy, adv_adjust_by, dx_adjust_by, dy_adjust_by)