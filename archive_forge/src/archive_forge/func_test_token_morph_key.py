import pytest
def test_token_morph_key(i_has):
    assert i_has[0].morph.key != 0
    assert i_has[1].morph.key != 0
    assert i_has[0].morph.key == i_has[0].morph.key
    assert i_has[0].morph.key != i_has[1].morph.key