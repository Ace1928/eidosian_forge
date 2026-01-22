from srsly.msgpack import unpackb
def testFixArray():
    check(b'\x92\x90\x91\x91\xc0', ((), ((None,),)))