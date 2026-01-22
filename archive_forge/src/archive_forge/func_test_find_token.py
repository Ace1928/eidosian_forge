from pyxnat import xpass
def test_find_token():
    print('Testing find_token')
    string = 'hello,world'
    assert xpass.find_token(',', string) == ('hello', 'world')
    assert xpass.find_token(' ', string) is None