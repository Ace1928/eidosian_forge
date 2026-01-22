import pytest
@pytest.mark.parametrize('text_poss,text', [("Robin's", 'Robin'), ("Alexis's", 'Alexis')])
def test_en_tokenizer_handles_poss_contraction(en_tokenizer, text_poss, text):
    tokens = en_tokenizer(text_poss)
    assert len(tokens) == 2
    assert tokens[0].text == text
    assert tokens[1].text == "'s"