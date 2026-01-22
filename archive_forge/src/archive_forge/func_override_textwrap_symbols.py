from .. import tests, utextwrap
def override_textwrap_symbols(testcase):
    testcase.overrideAttr(test_textwrap, 'TextWrapper', utextwrap.UTextWrapper)
    testcase.overrideAttr(test_textwrap, 'wrap', utextwrap.wrap)
    testcase.overrideAttr(test_textwrap, 'fill', utextwrap.fill)