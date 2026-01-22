from functools import wraps
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE
from ..validators import XMLSchema10, XMLSchema11, XsdGroup, \
@classmethod
def observed_builder(cls, builder):
    if isinstance(builder, type):

        class BuilderProxy(builder):

            def __init__(self, *args, **kwargs):
                super(BuilderProxy, self).__init__(*args, **kwargs)
                assert isinstance(self, XsdComponent)
                if not cls.is_dummy_component(self):
                    cls.components.append(self)
                else:
                    cls.dummy_components.append(self)
        BuilderProxy.__name__ = builder.__name__
        return BuilderProxy
    elif callable(builder):

        @wraps(builder)
        def builder_proxy(*args, **kwargs):
            obj = builder(*args, **kwargs)
            assert isinstance(obj, XsdComponent)
            if not cls.is_dummy_component(obj):
                cls.components.append(obj)
            else:
                cls.dummy_components.append(obj)
            return obj
        return builder_proxy