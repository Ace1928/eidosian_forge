from cirq.circuits._box_drawing_character_data import (
def test_chars():
    assert NORMAL_BOX_CHARS.char() is None
    assert NORMAL_BOX_CHARS.char(top=True, bottom=True) == '│'
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char() is None
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char(top=1, bottom=-1) == '╿'
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char(top=1, bottom=1) == '┃'
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char(top=-1, bottom=-1) == '│'
    assert box_draw_character(None, NORMAL_BOX_CHARS) is None
    assert box_draw_character(NORMAL_BOX_CHARS, BOLD_BOX_CHARS, top=-1, bottom=+1) == '╽'
    assert box_draw_character(BOLD_BOX_CHARS, NORMAL_BOX_CHARS, top=-1, bottom=+1) == '╿'
    assert box_draw_character(DOUBLED_BOX_CHARS, NORMAL_BOX_CHARS, left=-1, bottom=+1) == '╕'
    assert box_draw_character(NORMAL_BOX_CHARS, DOUBLED_BOX_CHARS, left=-1, bottom=+1) == '╖'