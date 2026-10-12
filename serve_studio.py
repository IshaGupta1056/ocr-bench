"""
serve_studio.py
Local server for OCR Bench Data Studio web application.
Serves the project directory so both studio/ and dataset/ assets are accessible,
and automatically opens http://localhost:8080/studio/ in your default browser.
"""

import os
import sys
import webbrowser
import http.server
import socketserver
from pathlib import Path

PORT = 8080
PROJECT_ROOT = Path(__file__).resolve().parent


class StudioHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler redirecting root to /studio/."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/studio/")
            self.end_headers()
            return
        super().do_GET()


def run_studio(port: int = PORT):
    os.chdir(PROJECT_ROOT)
    
    # Try preferred port, find next available if in use
    current_port = port
    max_tries = 10
    httpd = None

    for i in range(max_tries):
        try:
            httpd = socketserver.TCPServer(("", current_port), StudioHTTPHandler)
            break
        except OSError:
            current_port += 1

    if not httpd:
        print(f"[!] Could not bind to ports between {port} and {current_port}")
        sys.exit(1)

    url = f"http://localhost:{current_port}/studio/"
    print("=" * 65)
    print(f"[*] OCR Bench Data Studio running at: {url}")
    print(f"[*] Serving Gold Benchmark Dataset from: {PROJECT_ROOT / 'dataset'}")
    print("[*] Press Ctrl+C in terminal to stop.")
    print("=" * 65)

    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Server stopped successfully.")


if __name__ == "__main__":
    run_studio()
