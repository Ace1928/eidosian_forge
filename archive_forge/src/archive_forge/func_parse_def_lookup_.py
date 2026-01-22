import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_def_lookup_(self):
    assert self.is_cur_keyword_('DEF_LOOKUP')
    location = self.cur_token_location_
    name = self.expect_string_()
    if not name[0].isalpha():
        raise VoltLibError('Lookup name "%s" must start with a letter' % name, location)
    if self.lookups_.resolve(name) is not None:
        raise VoltLibError('Lookup "%s" already defined, lookup names are case insensitive' % name, location)
    process_base = True
    if self.next_token_ == 'PROCESS_BASE':
        self.advance_lexer_()
    elif self.next_token_ == 'SKIP_BASE':
        self.advance_lexer_()
        process_base = False
    process_marks = True
    mark_glyph_set = None
    if self.next_token_ == 'PROCESS_MARKS':
        self.advance_lexer_()
        if self.next_token_ == 'MARK_GLYPH_SET':
            self.advance_lexer_()
            mark_glyph_set = self.expect_string_()
        elif self.next_token_ == 'ALL':
            self.advance_lexer_()
        elif self.next_token_ == 'NONE':
            self.advance_lexer_()
            process_marks = False
        elif self.next_token_type_ == Lexer.STRING:
            process_marks = self.expect_string_()
        else:
            raise VoltLibError('Expected ALL, NONE, MARK_GLYPH_SET or an ID. Got %s' % self.next_token_type_, location)
    elif self.next_token_ == 'SKIP_MARKS':
        self.advance_lexer_()
        process_marks = False
    direction = None
    if self.next_token_ == 'DIRECTION':
        self.expect_keyword_('DIRECTION')
        direction = self.expect_name_()
        assert direction in ('LTR', 'RTL')
    reversal = None
    if self.next_token_ == 'REVERSAL':
        self.expect_keyword_('REVERSAL')
        reversal = True
    comments = None
    if self.next_token_ == 'COMMENTS':
        self.expect_keyword_('COMMENTS')
        comments = self.expect_string_().replace('\\n', '\n')
    context = []
    while self.next_token_ in ('EXCEPT_CONTEXT', 'IN_CONTEXT'):
        context = self.parse_context_()
    as_pos_or_sub = self.expect_name_()
    sub = None
    pos = None
    if as_pos_or_sub == 'AS_SUBSTITUTION':
        sub = self.parse_substitution_(reversal)
    elif as_pos_or_sub == 'AS_POSITION':
        pos = self.parse_position_()
    else:
        raise VoltLibError('Expected AS_SUBSTITUTION or AS_POSITION. Got %s' % as_pos_or_sub, location)
    def_lookup = ast.LookupDefinition(name, process_base, process_marks, mark_glyph_set, direction, reversal, comments, context, sub, pos, location=location)
    self.lookups_.define(name, def_lookup)
    return def_lookup