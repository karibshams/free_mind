import os
from openai import OpenAI
from prompts import SYSTEM_PROMPT_PREDICT


class FearForecast:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def _call_ai(self, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_PREDICT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def generate_forecast(self, fear: str, belief_strength: int) -> dict:
        """Generate AI prediction + tip based on user's fear input."""
        prompt = f"Fear: {fear}\nBelief strength: {belief_strength}%"
        response = self._call_ai(prompt)
        return {"fear": fear, "belief_strength": belief_strength, "ai_response": response}

    def generate_insight(self, fear: str, prediction: str, outcome: str) -> dict:
        """Generate insight after user logs outcome (It Was Fine / It Happened)."""
        prompt = (
            f"Original fear: {fear}\n"
            f"AI prediction: {prediction}\n"
            f"Actual outcome: {outcome}"
        )
        response = self._call_ai(prompt)
        return {"outcome": outcome, "insight": response}
