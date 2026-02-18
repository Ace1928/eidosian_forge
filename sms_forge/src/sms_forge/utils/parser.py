import re
from typing import Optional

def extract_code(text: str) -> Optional[str]:
    """
    Extract verification codes (4-8 digits) from message body.
    Handles common patterns like 'Your code is 123456' or '123456 is your code'.
    """
    # Look for 4-8 digit numbers that are not part of a larger number/string
    patterns = [
        r"(?i)code[:\s]+(\d{4,8})",
        r"(?i)is[:\s]+(\d{4,8})",
        r"(?i)verification[:\s]+(\d{4,8})",
        r"(\d{4,8})[\s\.]" # Just a lone 4-8 digit number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # Fallback: find any 6 digit sequence
    match = re.search(r"\b\d{6}\b", text)
    return match.group(0) if match else None
