import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_str_schema():
    assert str(schema).strip() == dedent('\n        schema {\n          query: RootQuery\n        }\n\n        type MyNode implements Node {\n          """The ID of the object"""\n          id: ID!\n          name: String\n        }\n\n        """An object with an ID"""\n        interface Node {\n          """The ID of the object"""\n          id: ID!\n        }\n\n        type MyOtherNode implements Node {\n          """The ID of the object"""\n          id: ID!\n          shared: String\n          somethingElse: String\n          extraField: String\n        }\n\n        type RootQuery {\n          first: String\n          node(\n            """The ID of the object"""\n            id: ID!\n          ): Node\n          onlyNode(\n            """The ID of the object"""\n            id: ID!\n          ): MyNode\n          onlyNodeLazy(\n            """The ID of the object"""\n            id: ID!\n          ): MyNode\n        }\n        ').strip()