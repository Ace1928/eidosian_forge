from yaql.language import exceptions
import yaql.tests
def test_generate_many(self):
    friends = {'John': ['Jim'], 'Jim': ['Jay', 'Jax'], 'Jax': ['John', 'Jacob', 'Jonathan'], 'Jacob': ['Jonathan', 'Jenifer']}
    self.assertEqual(['John', 'Jim', 'Jay', 'Jax', 'Jacob', 'Jonathan', 'Jenifer'], self.eval('generateMany(John, $data.get($, []), decycle => true)', friends))
    self.assertEqual(['John', 'Jim', 'Jay', 'Jax', 'Jacob', 'Jonathan', 'Jenifer'], self.eval('generateMany(John, $data.get($, []), decycle => true, depthFirst => true)', friends))
    self.assertEqual(['Jay'], self.eval('generateMany(Jay, $data.get($, []))', friends))
    self.assertEqual(['JAX', 'JOHN', 'JACOB', 'JONATHAN', 'JIM', 'JENIFER', 'JAY'], self.eval('generateMany(Jax, $data.get($, []), $.toUpper(), decycle => true)', friends))