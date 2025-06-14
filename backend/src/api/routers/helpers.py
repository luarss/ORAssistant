import os
from typing import Any
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from ..models.response_model import SuggestedQuestionInput
from ...prompts.prompt_templates import suggested_questions_prompt_template
from openai import OpenAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta", api_key=GEMINI_API_KEY
)

router = APIRouter(prefix="/helpers", tags=["helpers"])


# Main Router
@router.post("/suggestedQuestions")
async def get_suggested_questions(
    suggested_question_input: SuggestedQuestionInput,
) -> Any:
    full_prompt = suggested_questions_prompt_template.format(
        latest_question=suggested_question_input.latest_question,
        assistant_answer=suggested_question_input.assistant_answer,
    )

    try:
        response = client.chat.completions.create(
            model="gemini-1.5-flash-latest",
            messages=[{"role": "user", "content": full_prompt}],
        )
        # To match with the frontend expected schema
        return {
            "candidates": [
                {"content": {"parts": [{"text": response.choices[0].message.content}]}}
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Failed to get suggested questions: " + str(e)
        )
