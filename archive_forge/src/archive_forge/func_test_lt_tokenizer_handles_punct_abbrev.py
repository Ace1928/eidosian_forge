import pytest
@pytest.mark.parametrize('text,length', [('177R Parodų rūmai–Ozo g. nuo vasario 18 d. bus skelbiamas interneto tinklalapyje.', 17), ('ISM universiteto doc. dr. Ieva Augutytė-Kvedaravičienė pastebi, kad tyrimais nustatyti elgesio pokyčiai.', 18)])
def test_lt_tokenizer_handles_punct_abbrev(lt_tokenizer, text, length):
    tokens = lt_tokenizer(text)
    assert len(tokens) == length