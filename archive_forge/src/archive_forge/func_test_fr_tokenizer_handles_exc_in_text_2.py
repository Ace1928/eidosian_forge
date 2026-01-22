import pytest
def test_fr_tokenizer_handles_exc_in_text_2(fr_tokenizer):
    text = 'Cette après-midi, je suis allé dans un restaurant italo-mexicain.'
    tokens = fr_tokenizer(text)
    assert len(tokens) == 11
    assert tokens[1].text == 'après-midi'
    assert tokens[9].text == 'italo-mexicain'