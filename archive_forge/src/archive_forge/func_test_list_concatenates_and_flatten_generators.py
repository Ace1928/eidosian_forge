from yaql.language import exceptions
import yaql.tests
def test_list_concatenates_and_flatten_generators(self):
    generator_func = lambda: (i for _ in range(2) for i in range(3))
    self.context['$seq1'] = generator_func()
    self.context['$seq2'] = generator_func()
    self.assertEqual([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], self.eval('list($seq1, $seq2)'))