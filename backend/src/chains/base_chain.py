from typing import Optional, Union, Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from ..vectorstores.faiss import FAISSVectorDatabase


class BaseChain:
    def __init__(
        self,
        llm_model: Optional[
            Union[ChatGoogleGenerativeAI, ChatVertexAI, ChatOllama]
        ] = None,
        vector_db: Optional[FAISSVectorDatabase] = None,
        prompt_template_str: Optional[str] = None,
    ):
        self.llm_model = llm_model
        if prompt_template_str is not None:
            self.prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

        self.vector_db: Optional[FAISSVectorDatabase] = vector_db
        self.llm_chain: Any = None

    def create_llm_chain(self) -> None:
        self.llm_chain = self.prompt_template | self.llm_model | StrOutputParser()  # type: ignore

    def get_llm_chain(self) -> Any:
        if self.llm_chain is None:
            self.create_llm_chain()
        return self.llm_chain
