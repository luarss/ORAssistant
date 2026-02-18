"""
Custom DeepEvalLLM wrapper using Google Gemini API (google.genai).
DEPRECATED: This file uses deepeval and is kept for reference only.
"""

import os
from typing import Any, Optional, Type

from google import genai
from google.genai import types
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel


class Response(BaseModel):
    content: str


class GoogleGeminiLangChain(DeepEvalBaseLLM):
    """Class that implements Google Gemini API for DeepEval"""

    def __init__(self, model_name, *args, **kwargs):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        super().__init__(model_name, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        return self.client.models

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        if schema is not None:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return response.parsed, 0
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text, 0

    async def a_generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        if schema is not None:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return response.parsed, 0
        else:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text, 0

    def get_model_name(self):
        return self.model_name or "model-not-specified"
