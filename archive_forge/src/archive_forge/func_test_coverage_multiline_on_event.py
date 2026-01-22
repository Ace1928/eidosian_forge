import pytest
import os
def test_coverage_multiline_on_event():
    from kivy.lang.builder import Builder
    cov = coverage.Coverage(source=[os.path.dirname(__file__)])
    cov.start()
    fname = os.path.join(os.path.dirname(__file__), 'coverage_lang.kv')
    try:
        widget = Builder.load_file(fname)
        widget.children[0].y = 65
    finally:
        cov.stop()
    Builder.unload_file(fname)
    _, statements, missing, _ = cov.analysis(fname)
    assert set(statements) == kv_statement_lines
    assert set(missing) == {4}