from .roundtrip import YAML
def test_decorator_explicit(self):
    from srsly.ruamel_yaml import yaml_object
    yml = YAML()

    @yaml_object(yml)
    class User3(object):
        yaml_tag = u'!USER'

        def __init__(self, name, age):
            self.name = name
            self.age = age

        @classmethod
        def to_yaml(cls, representer, node):
            return representer.represent_scalar(cls.yaml_tag, u'{.name}-{.age}'.format(node, node))

        @classmethod
        def from_yaml(cls, constructor, node):
            return cls(*node.value.split('-'))
    ys = '\n        - !USER Anthon-18\n        '
    d = yml.load(ys)
    yml.dump(d, compare=ys)