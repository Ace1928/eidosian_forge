from .roundtrip import YAML
def test_register_0_rt(self):
    yaml = YAML()
    yaml.register_class(User0)
    ys = '\n        - !User0\n          name: Anthon\n          age: 18\n        '
    d = yaml.load(ys)
    yaml.dump(d, compare=ys, unordered_lines=True)