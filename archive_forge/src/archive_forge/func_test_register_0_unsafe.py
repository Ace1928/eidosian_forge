from .roundtrip import YAML
def test_register_0_unsafe(self):
    yaml = YAML(typ='unsafe')
    yaml.register_class(User0)
    ys = '\n        - !User0 {age: 18, name: Anthon}\n        '
    d = yaml.load(ys)
    yaml.dump(d, compare=ys)