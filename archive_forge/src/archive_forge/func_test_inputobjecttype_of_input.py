from graphql import Undefined
from ..argument import Argument
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..objecttype import ObjectType
from ..scalars import Boolean, String
from ..schema import Schema
from ..unmountedtype import UnmountedType
from ... import NonNull
def test_inputobjecttype_of_input():

    class Child(InputObjectType):
        first_name = String()
        last_name = String()

        @property
        def full_name(self):
            return f'{self.first_name} {self.last_name}'

    class Parent(InputObjectType):
        child = InputField(Child)

    class Query(ObjectType):
        is_child = Boolean(parent=Parent())

        def resolve_is_child(self, info, parent):
            return isinstance(parent.child, Child) and parent.child.full_name == 'Peter Griffin'
    schema = Schema(query=Query)
    result = schema.execute('query basequery {\n        isChild(parent: {child: {firstName: "Peter", lastName: "Griffin"}})\n    }\n    ')
    assert not result.errors
    assert result.data == {'isChild': True}