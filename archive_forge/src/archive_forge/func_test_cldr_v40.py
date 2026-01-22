import langcodes
def test_cldr_v40():
    en = langcodes.get('en')
    assert en.language_name('dsb') == 'engelšćina'