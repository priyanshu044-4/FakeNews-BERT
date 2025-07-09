import requests

def get_reasoning(news_text):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:12b",
                "prompt": f"""Analyze this news and decide if it's REAL or FAKE. Keep it short (max 4 lines). End with a clear verdict.

News: "{news_text}"
"""
            },
            stream=True
        )

        content = ""
        for line in response.iter_lines():
            if line:
                try:
                    part = line.decode("utf-8")
                    json_obj = eval(part)  # 👈 or use json.loads() if it's standard
                    content += json_obj.get("response", "")
                except Exception as e:
                    continue

        return content.strip() if content else "⚠️ No response from model."

    except Exception as e:
        return f"⚠️ Error talking to LLM: {e}"
