import http.server
import socketserver
import webbrowser

PORT = 7654
FILE = "index.html"
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"Serving at {url}")
    webbrowser.open(url)
    httpd.serve_forever()