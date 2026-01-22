from netaddr.strategy import eui48
def test_strategy_eui48():
    b = '00000000-00001111-00011111-00010010-11100111-00110011'
    i = 64945841971
    t = (0, 15, 31, 18, 231, 51)
    s = '00-0F-1F-12-E7-33'
    p = b'\x00\x0f\x1f\x12\xe73'
    assert eui48.bits_to_int(b) == i
    assert eui48.int_to_bits(i) == b
    assert eui48.int_to_str(i) == s
    assert eui48.str_to_int(s) == i
    assert eui48.int_to_words(i) == t
    assert eui48.words_to_int(t) == i
    assert eui48.words_to_int(list(t)) == i
    assert eui48.int_to_packed(i) == p
    assert eui48.packed_to_int(p) == i