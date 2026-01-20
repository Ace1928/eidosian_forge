import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

const run = (command, args) => {
  const result = spawnSync(command, args, { stdio: "inherit", shell: true });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
};

if (!existsSync("node_modules")) {
  run("npm", ["install"]);
}

run("npm", ["run", "build"]);
run("npm", ["run", "serve"]);
