import pytest
@pytest.mark.parametrize('word,norm', [('SUMMER', 'サマー'), ('食べ物', '食べ物'), ('綜合', '総合'), ('コンピュータ', 'コンピューター')])
def test_ja_lemmatizer_norm(ja_tokenizer, word, norm):
    test_norm = ja_tokenizer(word)[0].norm_
    assert test_norm == norm