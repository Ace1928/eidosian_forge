from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
def test_mutation_custom_output_type():

    class User(ObjectType):
        name = String()

    class CreateUser(Mutation):

        class Arguments:
            name = String()
        Output = User

        def mutate(self, info, name):
            return User(name=name)
    field = CreateUser.Field()
    assert field.type == User
    assert field.args == {'name': Argument(String)}
    resolved = field.resolver(None, None, name='Peter')
    assert isinstance(resolved, User)
    assert resolved.name == 'Peter'