def test_sv_tokenizer_handles_long_text(sv_tokenizer):
    text = 'Det var så härligt ute på landet. Det var sommar, majsen var gul, havren grön,\nhöet var uppställt i stackar nere vid den gröna ängen, och där gick storken på sina långa,\nröda ben och snackade engelska, för det språket hade han lärt sig av sin mor.\n\nRunt om åkrar och äng låg den stora skogen, och mitt i skogen fanns djupa sjöar; jo, det var verkligen trevligt ute på landet!'
    tokens = sv_tokenizer(text)
    assert len(tokens) == 86