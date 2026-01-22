from typing import List
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.llms.openai import OpenAI
from langchain_community.tools.vectorstore.tool import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.vectorstores import VectorStore
from langchain.tools import BaseTool
def get_tools(self) -> List[BaseTool]:
    """Get the tools in the toolkit."""
    tools: List[BaseTool] = []
    for vectorstore_info in self.vectorstores:
        description = VectorStoreQATool.get_description(vectorstore_info.name, vectorstore_info.description)
        qa_tool = VectorStoreQATool(name=vectorstore_info.name, description=description, vectorstore=vectorstore_info.vectorstore, llm=self.llm)
        tools.append(qa_tool)
    return tools