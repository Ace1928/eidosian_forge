import pytest
@pytest.mark.parametrize('text,length', [('ቻንስለር ጀርመን ኣንገላ መርከል፧', 5), ('“ስድራቤታት፧”', 4), ('ኣብ እዋን በዓላት ልደት ስድራቤታት ክተኣኻኸባ ዝፍቀደለን`ኳ እንተኾነ።', 9), ('ብግምት 10ኪ.ሜ. ጎይዩ።', 6), ('ኣብ ዝሓለፈ 24 ሰዓታት...', 5)])
def test_ti_tokenizer_handles_cnts(ti_tokenizer, text, length):
    tokens = ti_tokenizer(text)
    assert len(tokens) == length