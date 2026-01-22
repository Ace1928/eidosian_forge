import copy
from ..argument import Argument
from ..definitions import GrapheneGraphQLType
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_defines_a_mutation_schema():
    blog_schema = Schema(Query, mutation=Mutation)
    assert blog_schema.mutation == Mutation
    assert blog_schema.graphql_schema.mutation_type.graphene_type == Mutation
    write_mutation = Mutation._meta.fields['write_article']
    assert write_mutation.type == Article
    assert write_mutation.type._meta.name == 'Article'