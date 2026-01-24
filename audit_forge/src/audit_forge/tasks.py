import re
from pathlib import Path
from typing import Optional
from eidosian_core import eidosian


class IdempotentTaskManager:
    """
    Manages Markdown-based TODO and Roadmap files safely.
    Focuses on non-destructive, additive, and idempotent operations.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def _read_content(self) -> str:
        if not self.file_path.exists():
            return ""
        return self.file_path.read_text()

    def _write_content(self, content: str):
        self.file_path.write_text(content)

    @eidosian()
    def add_task(
        self, section: str, task_text: str, task_id: Optional[str] = None
    ) -> bool:
        """
        Adds a task to a specific section if it doesn't already exist.
        """
        content = self._read_content()
        new_line = (
            f"- [ ] **{task_id}** {task_text}" if task_id else f"- [ ] {task_text}"
        )

        if new_line in content:
            return False  # Idempotent: already exists

        lines = content.splitlines()
        found_section = False
        new_lines = []

        for line in lines:
            new_lines.append(line)
            if section.lower() in line.lower() and (
                line.startswith("#") or line.startswith("##")
            ):
                found_section = True

        if found_section:
            # Append to the end of the section (just before the next header or end of file)
            # This is a simplified version; in reality, we'd find the exact insertion point.
            new_lines.append(new_line)
        else:
            # Create section if missing
            new_lines.extend(["", f"## {section}", new_line])

        self._write_content("\n".join(new_lines) + "\n")
        return True

    @eidosian()
    def cross_off_task(self, task_identifier: str) -> bool:
        """
        Crosses off a task (changes [ ] to [x]) based on a string match or ID.
        """
        content = self._read_content()
        # Regex to find the task line and ensure it's not already crossed off
        pattern = re.compile(rf"- \[ \](.*{re.escape(task_identifier)}.*)")
        if not pattern.search(content):
            return False

        new_content = pattern.sub(r"- [x]\1", content)
        self._write_content(new_content)
        return True

    @eidosian()
    def extend_task_details(self, task_identifier: str, details: str) -> bool:
        """
        Adds nested details below a task.
        """
        content = self._read_content()
        lines = content.splitlines()
        new_lines = []
        modified = False

        for line in lines:
            new_lines.append(line)
            if task_identifier in line and "- [" in line and not modified:
                indent = "    "
                new_lines.append(f"{indent}- {details}")
                modified = True

        if modified:
            self._write_content("\n".join(new_lines) + "\n")
        return modified
