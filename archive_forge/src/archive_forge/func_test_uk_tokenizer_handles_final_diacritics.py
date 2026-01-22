import pytest
def test_uk_tokenizer_handles_final_diacritics(uk_tokenizer):
    text = 'Хлібі́в не було́. Хлібі́в не було́.'
    tokens = uk_tokenizer(text)
    assert tokens[2].text == 'було́'
    assert tokens[3].text == '.'