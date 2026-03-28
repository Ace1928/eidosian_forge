# 💎 ATLAS FORGE 🌍

> _"The map is the territory, rendered for the enlightened mind."_

## 🧠 IDENTITY & PURPOSE
The **Atlas Forge** is the primary visualization and monitoring hub of the Eidosian ecosystem. It provides high-fidelity, real-time telemetry, documentation browsing, and system orchestration interfaces.

## 📜 PRIME DIRECTIVES
1.  **High-Fidelity Telemetry**: System and Forge metrics must be accurate and performant.
2.  **Modular Aesthetics**: UI components should be modular and follow the "Velvet Beef" design language.
3.  **Recursive Observability**: Atlas should monitor itself and provide paths for its own improvement.
4.  **Integrated Navigation**: Seamlessly bridge the gap between raw data, documentation, and operational control.

## 🏗️ ARCHITECTURE
*   **Backend**: FastAPI-powered modular router system.
*   **Frontend**: Jinja2 SSR with custom Eidosian CSS.
*   **Observability**: Integrated PTY for shell access and real-time log tailing.
*   **Knowledge Bridge**: Direct indexing of `doc_forge` and `file_forge` outputs.

## 🛠️ OPERATIONAL STANDARDS
*   **Modularization**: Avoid large monolithic files (keep < 500 LOC per module).
*   **Safety**: Ensure all system-level calls (e.g., `psutil`, `/proc`) are robustly wrapped for Android/Termux.
*   **Performance**: Use async handlers and caching where appropriate to minimize latency.
