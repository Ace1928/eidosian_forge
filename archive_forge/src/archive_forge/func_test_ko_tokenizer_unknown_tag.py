import pytest
@pytest.mark.issue(10535)
def test_ko_tokenizer_unknown_tag(ko_tokenizer):
    tokens = ko_tokenizer('미닛 리피터')
    assert tokens[1].pos_ == 'X'