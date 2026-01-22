import pytest
@pytest.mark.parametrize('text,length', [('ગુજરાતીઓ ખાવાના શોખીન માનવામાં આવે છે', 6), ('ખેતરની ખેડ કરવામાં આવે છે.', 5)])
def test_gu_tokenizer_handles_cnts(gu_tokenizer, text, length):
    tokens = gu_tokenizer(text)
    assert len(tokens) == length