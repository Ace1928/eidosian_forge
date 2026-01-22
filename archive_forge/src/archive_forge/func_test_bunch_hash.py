import os
import pytest
from pkg_resources import resource_filename as pkgrf
from ....utils.filemanip import md5
from ... import base as nib
def test_bunch_hash():
    json_pth = pkgrf('nipype', os.path.join('testing', 'data', 'realign_json.json'))
    b = nib.Bunch(infile=json_pth, otherthing='blue', yat=True)
    newbdict, bhash = b._get_bunch_hash()
    assert bhash == 'd1f46750044c3de102efc847720fc35f'
    jshash = md5()
    with open(json_pth, 'r') as fp:
        jshash.update(fp.read().encode('utf-8'))
    assert newbdict['infile'][0][1] == jshash.hexdigest()
    assert newbdict['yat'] is True