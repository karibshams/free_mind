import os
from openai import OpenAI
from prompts import SYSTEM_PROMPT_CHAT


class AICoach:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.history = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]

    def chat(self, user_message: str) -> str:
        """Send a message and get a response. Maintains conversation history."""
        self.history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=self.history,
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Clear conversation history for a new session."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
