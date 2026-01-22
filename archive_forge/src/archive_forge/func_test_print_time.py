import re
from joblib.logger import PrintTime
def test_print_time(tmpdir, capsys):
    logfile = tmpdir.join('test.log').strpath
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    out_printed_text, err_printed_text = capsys.readouterr()
    match = 'Foo: 0\\..s, 0\\..min\\nFoo: 0\\..s, 0..min\\nFoo: ' + '.\\..s, 0..min\\n'
    if not re.match(match, err_printed_text):
        raise AssertionError('Excepted %s, got %s' % (match, err_printed_text))