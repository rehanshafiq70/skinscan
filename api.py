from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic, base64, json, re

app = Flask(__name__)
CORS(app, origins="*")

SYSTEM_PROMPT = """You are SkinScan AI, expert dermatology analysis assistant.
Analyze this skin lesion image. Return ONLY valid JSON, no extra text:
{
  "binary_result": "BENIGN",
  "confidence": 87.5,
  "risk_level": "LOW",
  "multi_class_probabilities": {
    "MEL": 0.05, "NV": 0.60, "BCC": 0.08, "AKIEC": 0.07,
    "BKL": 0.12, "DF": 0.04, "VASC": 0.02, "SCC": 0.02
  },
  "abcde_analysis": {
    "asymmetry": {"score": 1, "desc": "Symmetric lesion, low concern"},
    "border": {"score": 1, "desc": "Well-defined regular borders"},
    "color": {"score": 1, "desc": "Uniform brown pigmentation"},
    "diameter": {"score": 2, "desc": "Approximately 5mm diameter"},
    "evolution": {"score": 1, "desc": "No reported recent changes"}
  },
  "clinical_recommendation": "Routine monitoring recommended. Annual skin check advised.",
  "urgent": false,
  "condition_summary": "Lesion consistent with benign melanocytic nevus.",
  "differential_diagnosis": ["Melanocytic Nevus", "Seborrheic Keratosis", "Dermatofibroma"]
}
Probabilities must sum to 1.0. Score 1=low, 2=medium, 3=high concern."""

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json or {}
        api_key = data.get('api_key', '').strip()
        image_b64 = data.get('image', '')
        patient = data.get('patient_info', {})
        if not api_key:
            return jsonify({"error": "No Anthropic API key provided"}), 400
        if not image_b64:
            return jsonify({"error": "No image data provided"}), 400
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        client = anthropic.Anthropic(api_key=api_key)
        ctx = f"Patient: {patient.get('name','Unknown')}, Age: {patient.get('age','?')}, Gender: {patient.get('gender','?')}"
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": ctx + "\nAnalyze this skin lesion image and return the JSON."}
            ]}]
        )
        raw = "".join(b.get("text","") for b in msg.content if b.type == "text")
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "AI response parse error", "raw": raw[:300]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "SkinScan AI Backend"})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
