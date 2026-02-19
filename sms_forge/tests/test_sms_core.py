import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sms_forge.core import SmsForge
from sms_forge.providers.termux import TermuxProvider
from sms_forge.utils.parser import extract_code

def test_extract_code():
    assert extract_code("Your code is 1234") == "1234"
    assert extract_code("Verification: 567890.") == "567890"
    assert extract_code("Use 999111 to login") == "999111"
    assert extract_code("No code here") is None

@pytest.mark.asyncio
async def test_sms_forge_auto_detection():
    with patch('sms_forge.providers.termux.TermuxProvider.is_available', return_value=True):
        forge = SmsForge()
        assert isinstance(forge.provider, TermuxProvider)

@pytest.mark.asyncio
async def test_termux_send_mock():
    with patch('asyncio.create_subprocess_exec') as mock_exec:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        
        provider = TermuxProvider()
        success = await provider.send("5551234", "Hello")
        assert success is True
        mock_exec.assert_called_once()
