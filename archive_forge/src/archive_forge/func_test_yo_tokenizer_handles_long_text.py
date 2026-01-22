import pytest
from spacy.lang.yo.lex_attrs import like_num
def test_yo_tokenizer_handles_long_text(yo_tokenizer):
    text = 'Àwọn ọmọ ìlú tí wọ́n ń ṣàmúlò ayélujára ti bẹ̀rẹ̀ ìkọkúkọ sórí àwòrán ààrẹ Nkurunziza nínú ìfẹ̀hónúhàn pẹ̀lú àmì ìdámọ̀: Nkurunziza àti Burundi:\n        Ọmọ ilé ẹ̀kọ́ gíga ní ẹ̀wọ̀n fún kíkọ ìkọkúkọ sí orí àwòrán Ààrẹ .\n        Bí mo bá ṣe èyí ní Burundi , ó ṣe é ṣe kí a fi mí sí àtìmọ́lé\n        Ìjọba Burundi fi akẹ́kọ̀ọ́bìnrin sí àtìmọ́lé látàrí ẹ̀sùn ìkọkúkọ sí orí àwòrán ààrẹ. A túwíìtì àwòrán ìkọkúkọ wa ní ìbánikẹ́dùn ìṣẹ̀lẹ̀ náà.\n        Wọ́n ní kí a dán an wò, kí a kọ nǹkan sí orí àwòrán ààrẹ  mo sì ṣe bẹ́ẹ̀. Mo ní ìgbóyà wípé ẹnikẹ́ni kò ní mú mi níbí.\n        Ìfòfinlíle mú àtakò'
    tokens = yo_tokenizer(text)
    assert len(tokens) == 121