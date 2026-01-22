import os
import pytest
from bs4 import (
@pytest.mark.skip(reason='html5lib-specific problems')
@pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-4818336571064320', 'clusterfuzz-testcase-minimized-bs4_fuzzer-4999465949331456', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5843991618256896', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6241471367348224', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6600557255327744', 'crash-0d306a50c8ed8bcd0785b67000fcd5dea1d33f08'])
def test_html5lib_parse_errors_without_css(self, filename):
    markup = self.__markup(filename)
    print(BeautifulSoup(markup, 'html5lib').encode())