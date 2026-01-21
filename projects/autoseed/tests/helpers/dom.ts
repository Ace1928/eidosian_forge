import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";

export const loadHudMarkup = (): void => {
  const html = readFileSync(resolve(cwd(), "src/index.html"), "utf8");
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  const bodyHtml = bodyMatch ? bodyMatch[1] : html;
  const sanitized = bodyHtml.replace(/<script[\s\S]*?<\/script>/gi, "");
  document.body.innerHTML = sanitized;
};
