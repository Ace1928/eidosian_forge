import pickle
import random
from netaddr import (
def test_eui_constructor():
    assert str(EUI('00-1B-77-49-54-FD')) == '00-1B-77-49-54-FD'
    assert str(EUI('00-1b-77-49-54-fd')) == '00-1B-77-49-54-FD'
    assert str(EUI('0:1b:77:49:54:fd')) == '00-1B-77-49-54-FD'
    assert str(EUI('001b:7749:54fd')) == '00-1B-77-49-54-FD'
    assert str(EUI('1b:7749:54fd')) == '00-1B-77-49-54-FD'
    assert str(EUI('1B:7749:54FD')) == '00-1B-77-49-54-FD'
    assert str(EUI('001b774954fd')) == '00-1B-77-49-54-FD'
    assert str(EUI('01B774954FD')) == '00-1B-77-49-54-FD'
    assert str(EUI('001B77:4954FD')) == '00-1B-77-49-54-FD'
    eui = EUI('00-90-96-AF-CC-39')
    assert eui == EUI('0-90-96-AF-CC-39')
    assert eui == EUI('00-90-96-af-cc-39')
    assert eui == EUI('00:90:96:AF:CC:39')
    assert eui == EUI('00:90:96:af:cc:39')
    assert eui == EUI('0090-96AF-CC39')
    assert eui == EUI('0090:96af:cc39')
    assert eui == EUI('009096-AFCC39')
    assert eui == EUI('009096:AFCC39')
    assert eui == EUI('009096AFCC39')
    assert eui == EUI('009096afcc39')
    assert EUI('01-00-00-00-00-00') == EUI('010000000000')
    assert EUI('01-00-00-00-00-00') == EUI('10000000000')
    assert EUI('01-00-00-01-00-00') == EUI('010000:010000')
    assert EUI('01-00-00-01-00-00') == EUI('10000:10000')