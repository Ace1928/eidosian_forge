import asyncio
import os

import pytest

from moltbook_forge.client import MoltbookClient

INTEGRATION_ENABLED = os.getenv("MOLTBOOK_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not INTEGRATION_ENABLED,
    reason="Set MOLTBOOK_INTEGRATION=1 to run real API checks.",
)

@pytest.mark.asyncio
async def test_real_api_connectivity():
    """Verify we can reach the real Moltbook API."""
    client = MoltbookClient()
    print(f"\nConnecting to: {client.base_url}")
    
    try:
        # 1. Test profile fetch
        me = await client.get_me()
        if me:
            print(f"Successfully authenticated as: @{me.username}")
        else:
            print("Failed to fetch profile (me is None).")

        # 2. Test post fetch
        posts = await client.get_posts(limit=5)
        print(f"Fetched {len(posts)} real posts.")
        for p in posts:
            author = getattr(p, 'author', 'Unknown')
            content = getattr(p, 'content', '')
            print(f" - [{p.id}] @{author}: {content[:30]}...")
            
    except Exception as e:
        print(f"Error during API test: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_real_api_connectivity())
