#!/usr/bin/env python3
import sys
import json
import re
from pathlib import Path
from typing import Optional

# Add necessary paths for llm_forge and ollama_forge
FORGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORGE_ROOT / "llm_forge" / "src"))
sys.path.insert(0, str(FORGE_ROOT / "ollama_forge" / "src"))
from llm_forge.core.manager import ModelManager

def extract_profile_data(file_path: Path, llm_model: Optional[str] = None, max_tokens: Optional[int] = None):
    file_content = file_path.read_text()
    prompt = (
        'Extract the following from the provided text into a JSON object: '
        '"name", "email", "github", "profession", "interests" (as a list of strings), '
        '"skills" (as a list of strings), "personal_traits" (as a list of strings), '
        '"projects" (as a list of strings), "medium_goals" (as a list of strings). '
        'Only extract what is explicitly stated.\n\nText:\n'
    ) + file_content
    
    manager = ModelManager(default_ollama_timeout=3600.0)
    result = manager.generate(prompt=prompt, model=llm_model, max_tokens=max_tokens, timeout=3600.0, json_mode=True)
    
    # If an LLMResponse object is returned, it implies a successful call.
    # Any errors would have raised an exception prior to this point.
    if result.text:
        # Check if the LLM output is wrapped in a Markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', result.text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = result.text # Assume it's plain JSON
            
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            return {"error": "LLM output was not valid JSON", "raw_output": result.text}
    else:
        return {"error": "LLM generation returned empty text."}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: extract_profile_data.py <path_to_about_me.md> [llm_model] [max_tokens]"}))
        sys.exit(1)
    
    md_path = Path(sys.argv[1])
    if not md_path.exists():
        print(json.dumps({"error": f"File not found at {md_path}"}))
        sys.exit(1)
    
    llm_model_arg = sys.argv[2] if len(sys.argv) > 2 else None
    max_tokens_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    data = extract_profile_data(md_path, llm_model=llm_model_arg, max_tokens=max_tokens_arg)
    print(json.dumps(data, indent=2))
