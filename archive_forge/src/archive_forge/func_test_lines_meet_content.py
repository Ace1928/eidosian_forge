import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def test_lines_meet_content():
    d = BlockDiagramDrawer()
    b = d.mutable_block(0, 0)
    b.content = 'long text\nwith multiple lines'
    b.left = '>'
    b.right = '<'
    b.top = 'v'
    b.bottom = '^'
    _assert_same_diagram(d.render(), '\nlong text<<<<<<<<<<\nwith multiple lines'[1:])
    b.horizontal_alignment = 0.5
    _assert_same_diagram(d.render(), '\n>>>>>long text<<<<<\nwith multiple lines'[1:])
    _assert_same_diagram(d.render(min_block_height=5), '\n         v\n         v\n>>>>>long text<<<<<\nwith multiple lines\n         ^'[1:])
    _assert_same_diagram(d.render(min_block_height=4), '\n         v\n>>>>>long text<<<<<\nwith multiple lines\n         ^'[1:])
    _assert_same_diagram(d.render(min_block_height=20, min_block_width=40), '\n                   v\n                   v\n                   v\n                   v\n                   v\n                   v\n                   v\n                   v\n                   v\n>>>>>>>>>>>>>>>long text<<<<<<<<<<<<<<<<\n          with multiple lines\n                   ^\n                   ^\n                   ^\n                   ^\n                   ^\n                   ^\n                   ^\n                   ^\n                   ^'[1:])
    _assert_same_diagram(d.render(min_block_height=21, min_block_width=41), '\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n>>>>>>>>>>>>>>>>long text<<<<<<<<<<<<<<<<\n           with multiple lines\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^'[1:])
    b.content = 'short text'
    _assert_same_diagram(d.render(min_block_height=21, min_block_width=41), '\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n>>>>>>>>>>>>>>>>short text<<<<<<<<<<<<<<<\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^'[1:])
    b.content = 'abc\ndef\nghi'
    _assert_same_diagram(d.render(min_block_height=21, min_block_width=41), '\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                    v\n                   abc\n>>>>>>>>>>>>>>>>>>>def<<<<<<<<<<<<<<<<<<<\n                   ghi\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^\n                    ^'[1:])