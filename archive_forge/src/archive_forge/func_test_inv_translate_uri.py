from pyxnat import uriutil
def test_inv_translate_uri():
    assert uriutil.inv_translate_uri('/assessors/out/resources/files') == '/assessors/out_resources/files'
    assert uriutil.inv_translate_uri('/assessors/out/resource/files') == '/assessors/out_resource/files'
    assert uriutil.inv_translate_uri('/assessors/in/resources/files') == '/assessors/in_resources/files'
    assert uriutil.inv_translate_uri('/assessors/in/resource/files') == '/assessors/in_resource/files'