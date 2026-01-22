import asyncio
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from importlib import util as importlib_util
from traitlets import Bool, default
from .html import HTMLExporter
def run_playwright(self, html):
    """Run playwright."""

    async def main(temp_file):
        """Run main playwright script."""
        args = ['--no-sandbox'] if self.disable_sandbox else []
        try:
            from playwright.async_api import async_playwright
        except ModuleNotFoundError as e:
            msg = 'Playwright is not installed to support Web PDF conversion. Please install `nbconvert[webpdf]` to enable.'
            raise RuntimeError(msg) from e
        if self.allow_chromium_download:
            cmd = [sys.executable, '-m', 'playwright', 'install', 'chromium']
            subprocess.check_call(cmd)
        playwright = await async_playwright().start()
        chromium = playwright.chromium
        try:
            browser = await chromium.launch(handle_sigint=False, handle_sigterm=False, handle_sighup=False, args=args)
        except Exception as e:
            msg = "No suitable chromium executable found on the system. Please use '--allow-chromium-download' to allow downloading one,or install it using `playwright install chromium`."
            await playwright.stop()
            raise RuntimeError(msg) from e
        page = await browser.new_page()
        await page.emulate_media(media='print')
        await page.wait_for_timeout(100)
        await page.goto(f'file://{temp_file.name}', wait_until='networkidle')
        await page.wait_for_timeout(100)
        pdf_params = {'print_background': True}
        if not self.paginate:
            dimensions = await page.evaluate('() => {\n                    const rect = document.body.getBoundingClientRect();\n                    return {\n                    width: Math.ceil(rect.width) + 1,\n                    height: Math.ceil(rect.height) + 1,\n                    }\n                }')
            width = dimensions['width']
            height = dimensions['height']
            pdf_params.update({'width': min(width, 200 * 72), 'height': min(height, 200 * 72)})
        pdf_data = await page.pdf(**pdf_params)
        await browser.close()
        await playwright.stop()
        return pdf_data
    pool = concurrent.futures.ThreadPoolExecutor()
    temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
    with temp_file:
        temp_file.write(html.encode('utf-8'))
    try:

        def run_coroutine(coro):
            """Run an internal coroutine."""
            loop = asyncio.ProactorEventLoop() if IS_WINDOWS else asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        pdf_data = pool.submit(run_coroutine, main(temp_file)).result()
    finally:
        os.unlink(temp_file.name)
    return pdf_data