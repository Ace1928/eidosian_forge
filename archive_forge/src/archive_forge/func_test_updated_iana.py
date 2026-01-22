import langcodes
def test_updated_iana():
    aqk = langcodes.get('aqk')
    assert aqk.language_name('en') == 'Aninka'