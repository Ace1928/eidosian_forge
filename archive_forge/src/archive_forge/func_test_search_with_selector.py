import yaql.tests
def test_search_with_selector(self):
    self.assertEqual('24.16 = 24(2-4) + 16(5-7)', self.eval("regex(`(\\d+)\\.?(\\d+)?`).search('aa24.16bb', $.value + ' = ' + $2.value + '(' + str($2.start) + '-' + str($2.end) + ') + ' + $3.value + '(' + str($3.start) + '-' + str($3.end) + ')')"))