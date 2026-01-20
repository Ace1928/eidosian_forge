import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join } from "node:path";

const PORT = Number.parseInt(process.env.PORT ?? "8080", 10);
const ROOT = new URL("../dist/", import.meta.url).pathname;

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml"
};

const resolvePath = async (url) => {
  const path = url === "/" ? "/index.html" : url;
  const filePath = join(ROOT, decodeURIComponent(path));
  try {
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) {
      return null;
    }
    return filePath;
  } catch {
    return null;
  }
};

const server = http.createServer(async (req, res) => {
  const filePath = await resolvePath(req.url ?? "/");
  if (!filePath) {
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not found");
    return;
  }

  const ext = extname(filePath);
  const contentType = MIME[ext] ?? "application/octet-stream";
  try {
    const data = await readFile(filePath);
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  } catch {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end("Failed to read file");
  }
});

server.listen(PORT, () => {
  console.log(`Autoseed running at http://localhost:${PORT}`);
});
