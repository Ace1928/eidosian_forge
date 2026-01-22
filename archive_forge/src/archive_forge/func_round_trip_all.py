import sys
import textwrap
from pathlib import Path
def round_trip_all(self, stream, **kw):
    from srsly.ruamel_yaml.compat import StringIO, BytesIO
    assert isinstance(stream, (srsly.ruamel_yaml.compat.text_type, str))
    lkw = kw.copy()
    if stream and stream[0] == '\n':
        stream = stream[1:]
    stream = textwrap.dedent(stream)
    data = list(srsly.ruamel_yaml.YAML.load_all(self, stream))
    outp = lkw.pop('outp', stream)
    lkw['stream'] = st = StringIO()
    srsly.ruamel_yaml.YAML.dump_all(self, data, **lkw)
    res = st.getvalue()
    if res != outp:
        diff(outp, res, 'input string')
    assert res == outp