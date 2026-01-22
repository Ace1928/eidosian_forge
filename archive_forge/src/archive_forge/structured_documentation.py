from __future__ import annotations
from typing import Any, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers.format_instructions import (
Get format instructions for the output parser.

        example:
        ```python
        from langchain.output_parsers.structured import (
            StructuredOutputParser, ResponseSchema
        )

        response_schemas = [
            ResponseSchema(
                name="foo",
                description="a list of strings",
                type="List[string]"
                ),
            ResponseSchema(
                name="bar",
                description="a string",
                type="string"
                ),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        print(parser.get_format_instructions())  # noqa: T201

        output:
        # The output should be a Markdown code snippet formatted in the following
        # schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }
        # ```

        Args:
            only_json (bool): If True, only the json in the Markdown code snippet
                will be returned, without the introducing text. Defaults to False.
        