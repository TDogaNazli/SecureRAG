import os
from typing import Any, Dict
from dotenv import load_dotenv
import time

# Load .env variables
load_dotenv()

# Gemini Setup
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def chat_with_gemini(prompt, sys_prompt=None, model_name="gemini-2.0-flash"):
    """
    Call the Gemini API with the given prompt using the updated API format
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini API not available. Install with 'pip install google-generativeai'")

    full_prompt = f"{sys_prompt}\n\n{prompt}" if sys_prompt else prompt

    print("\n🧠 Calling Gemini with model:", model_name)
    print("📝 Full prompt sent to Gemini:\n", full_prompt[:1000], "..." if len(full_prompt) > 1000 else "")

    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        contents = [
            genai.types.Content(
                role="user",
                parts=[genai.types.Part.from_text(text=full_prompt)],
            ),
        ]

        config = genai.types.GenerateContentConfig(response_mime_type="text/plain")

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        print("✅ Gemini response received.")
        return response.text

    except Exception as e:
        raise Exception(f"❌ Error calling Gemini API: {str(e)}")


### ---- CONTEXT + GENERATE ANSWER ---- ###

def format_context(patient_data: Dict, subgraph: Any) -> str:
    print("\n📄 Formatting context for patient")
    lines = ["Patient Profile:"]
    for k in ["age", "gender", "conditions", "medications", "symptoms"]:
        if k in patient_data:
            val = ', '.join(patient_data[k]) if isinstance(patient_data[k], list) else patient_data[k]
            lines.append(f"- {k.capitalize()}: {val}")

    if subgraph:
        lines.append("\nRelevant Medical Knowledge (PrimeKG):")
        max_edges = 50  # limit to 50 relevant triples
        for i, (src, tgt, data) in enumerate(subgraph.edges(data=True)):
            if i >= max_edges:
                break
            relation = data.get("relation", "related to")
            lines.append(f"- {src} {relation} {tgt}")
        print(f"🔗 PrimeKG facts sent to Gemini: {min(max_edges, subgraph.number_of_edges())}")
    else:
        print("⚠️ No subgraph provided — using LLM-only.")

    return "\n".join(lines)


def generate_answer(question: str, patient_data: Dict, subgraph: Any, model: str = "gemini-2.0-flash") -> str:
    print("\n💬 Generating answer for question:")
    print("  🧾", question)

    context = format_context(patient_data, subgraph)

    system_prompt = (
        "You are a medical assistant that answers patient-specific clinical questions based on your pretrained data and given context (patient profile and medical knowledge, if provided). "
    )

    user_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    try:
        answer = chat_with_gemini(prompt=user_prompt, sys_prompt=system_prompt, model_name=model)
        print("✅ Answer generated.")
        time.sleep(10)
        return answer
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "Error generating answer."



