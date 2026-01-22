import pytest
def test_lb_tokenizer_handles_exc_in_text(lb_tokenizer):
    text = "Mee 't ass net evident, d'Liewen."
    tokens = lb_tokenizer(text)
    assert len(tokens) == 9
    assert tokens[1].text == "'t"