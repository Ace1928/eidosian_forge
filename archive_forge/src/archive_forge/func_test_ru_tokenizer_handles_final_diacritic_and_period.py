from string import punctuation
import pytest
@pytest.mark.parametrize('text', ['РЕКОМЕНДУ́Я ПОДДА́ТЬ ЖАРУ́.САМОГО́ БАРГАМОТА', 'рекоменду̍я подда̍ть жару́.самого́ Баргамота'])
def test_ru_tokenizer_handles_final_diacritic_and_period(ru_tokenizer, text):
    tokens = ru_tokenizer(text)
    assert tokens[2].text.lower() == 'жару́.самого́'