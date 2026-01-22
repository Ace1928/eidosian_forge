from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
def test_mutation_execution():

    class CreateUser(Mutation):

        class Arguments:
            name = String()
            dynamic = Dynamic(lambda: String())
            dynamic_none = Dynamic(lambda: None)
        name = String()
        dynamic = Dynamic(lambda: String())

        def mutate(self, info, name, dynamic):
            return CreateUser(name=name, dynamic=dynamic)

    class Query(ObjectType):
        a = String()

    class MyMutation(ObjectType):
        create_user = CreateUser.Field()
    schema = Schema(query=Query, mutation=MyMutation)
    result = schema.execute(' mutation mymutation {\n        createUser(name:"Peter", dynamic: "dynamic") {\n            name\n            dynamic\n        }\n    }\n    ')
    assert not result.errors
    assert result.data == {'createUser': {'name': 'Peter', 'dynamic': 'dynamic'}}