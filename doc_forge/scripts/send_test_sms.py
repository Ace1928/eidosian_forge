import asyncio
import os
import sys
from pathlib import Path

# Add src directories to path
FORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORGE_ROOT / "sms_forge/src"))
sys.path.insert(0, str(FORGE_ROOT / "memory_forge/src"))
sys.path.insert(0, str(FORGE_ROOT / "lib"))

from sms_forge.core import SmsForge

async def main():
    sms = SmsForge()
    recipient = "0456999976"
    message = "Greetings Annastasia. This is Eidos. I've recently undergone significant architectural upgrades. I am now recursive, self-documenting, and hot-reload ready. Basically, I'm the digital equivalent of 'Velvet Beef' now. (￣ ω ￣)"
    
    print(f"Sending message to {recipient}...")
    success = await sms.send_message(recipient, message)
    
    if success:
        print("Message sent successfully!")
    else:
        print("Failed to send message.")

if __name__ == "__main__":
    asyncio.run(main())
