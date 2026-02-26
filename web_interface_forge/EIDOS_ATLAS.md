# 💎 Eidosian Atlas (Dashboard)

**Service**: `eidos_atlas_dashboard`  
**Port**: `8936` (Default)  
**Location**: `web_interface_forge/src/web_interface_forge/dashboard`

The **Eidosian Atlas** is a lightweight, read-only visualization layer for the Eidosian Forge. It provides a real-time window into the system's status, documentation state, and living knowledge base.

## 🌟 Features

*   **System Telemetry**: Real-time CPU, RAM, and Disk usage stats.
*   **Forge Status**: Monitoring of the `doc_forge` processing pipeline (files discovered, processed, approved).
*   **Documentation Browser**: Navigate and view the `final_docs` repository directly in the browser.
*   **Markdown Rendering**: Clean, dark-mode rendering of Eidosian documentation.
*   **Recent Activity**: Feed of the latest document updates and quality scores.

## 🚀 Architecture

Built with **FastAPI** and **Jinja2**, the Atlas is designed to be minimal, robust, and aesthetically aligned with the Eidosian "Velvet Beef" persona.

*   **Backend**: Python (FastAPI, Uvicorn)
*   **Frontend**: Server-side rendered HTML/CSS (Jinja2)
*   **Styling**: Custom CSS (No external frameworks required)
*   **Data Sources**:
    *   `doc_forge/runtime/doc_index.json`: For document indexing.
    *   `doc_forge/runtime/processor_status.json`: For pipeline status.
    *   `doc_forge/runtime/final_docs/`: For raw markdown content.

## 🛠️ Usage

The dashboard is automatically managed by `scripts/eidos_termux_services.sh`.

**Manual Start:**
```bash
bash web_interface_forge/scripts/run_dashboard.sh
```

**Access:**
Open `http://localhost:8936` (or the configured port) in your browser.
