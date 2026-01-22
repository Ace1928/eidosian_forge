import pytest
def test_fr_tokenizer_handles_title_3(fr_tokenizer):
    text = "Qu'est-ce que tu fais?"
    tokens = fr_tokenizer(text)
    assert len(tokens) == 7
    assert tokens[0].text == "Qu'"