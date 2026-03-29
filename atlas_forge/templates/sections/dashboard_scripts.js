function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function selectedServiceTarget() {
    return document.getElementById("service-target")?.value?.trim() || "";
}

let atlasShellSessionId = "";
let atlasShellPollHandle = null;

function setShellStatus(statusText, sessionId = atlasShellSessionId) {
    const statusEl = document.getElementById("shell-status");
    const sessionEl = document.getElementById("shell-session-id");
    if (statusEl) statusEl.textContent = statusText || "idle";
    if (sessionEl) sessionEl.textContent = sessionId || "none";
}

function appendShellOutput(chunk) {
    const output = document.getElementById("shell-output");
    if (!output || !chunk) return;
    output.textContent += chunk;
    output.scrollTop = output.scrollHeight;
}

async function startShellSession() {
    const params = new URLSearchParams({
        cwd: document.getElementById("shell-cwd")?.value || "eidosian_forge",
        cols: document.getElementById("shell-cols")?.value || "120",
        rows: document.getElementById("shell-rows")?.value || "28",
    });
    const res = await fetch(`/api/shell/start?${params.toString()}`, {method: "POST"});
    if (!res.ok) {
        appendShellOutput(`
[atlas] shell start failed: ${res.status}
`);
        return;
    }
    const payload = await res.json();
    atlasShellSessionId = payload.session_id || "";
    document.getElementById("shell-output").textContent = "";
    setShellStatus(payload.status || "running", atlasShellSessionId);
    if (atlasShellPollHandle) clearInterval(atlasShellPollHandle);
    atlasShellPollHandle = setInterval(readShellOutput, 1500);
    await readShellOutput();
    await refreshRuntimeServices();
}

async function readShellOutput() {
    if (!atlasShellSessionId) return;
    const res = await fetch(`/api/shell/read?session_id=${encodeURIComponent(atlasShellSessionId)}&max_bytes=16384`, {cache: "no-store"});
    if (!res.ok) return;
    const payload = await res.json();
    setShellStatus(payload.status || "running", payload.session_id || atlasShellSessionId);
    appendShellOutput(payload.output || "");
    if ((payload.status || "").toLowerCase() !== "running") {
        if (atlasShellPollHandle) clearInterval(atlasShellPollHandle);
        atlasShellPollHandle = null;
    }
}

async function sendShellInput() {
    if (!atlasShellSessionId) {
        await startShellSession();
    }
    const input = document.getElementById("shell-input");
    const value = input?.value || "";
    if (!value.trim()) return;
    const normalized = value.endsWith("\n") ? value : `${value}\n`;
    await fetch(`/api/shell/input?session_id=${encodeURIComponent(atlasShellSessionId)}&text=${encodeURIComponent(normalized)}`, {method: "POST"});
    input.value = "";
    await readShellOutput();
}


async function sendShellCommand(command) {
    const input = document.getElementById("shell-input");
    if (input) input.value = command;
    await sendShellInput();
}

async function stopShellSession() {
    if (!atlasShellSessionId) return;
    await fetch(`/api/shell/stop?session_id=${encodeURIComponent(atlasShellSessionId)}`, {method: "POST"});
    if (atlasShellPollHandle) clearInterval(atlasShellPollHandle);
    atlasShellPollHandle = null;
    setShellStatus("stopped", atlasShellSessionId);
    atlasShellSessionId = "";
    await refreshRuntimeServices();
}

async function refreshFileForge() {
    try {
        const res = await fetch("/api/runtime/file-forge", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const summary = payload.summary || {};
        const latest = payload.index_status || {};
        document.getElementById("file-forge-total-files").textContent = summary.total_files ?? 0;
        document.getElementById("file-forge-kind-count").textContent = (summary.by_kind || []).length;
        document.getElementById("file-forge-job-status").textContent = latest.status || "idle";
        document.getElementById("file-forge-total-files-table").textContent = summary.total_files ?? 0;
        document.getElementById("file-forge-total-links").textContent = summary.total_links ?? 0;
        document.getElementById("file-forge-total-relationships").textContent = summary.total_relationships ?? 0;
        document.getElementById("file-forge-duplicates").textContent = summary.duplicate_groups ?? 0;
        document.getElementById("file-forge-latest-job").textContent = latest.status || "idle";
        const body = document.getElementById("file-forge-recent-body");
        if (body) {
            body.innerHTML = (summary.recent_files || []).slice(0, 6).map((row) => `
                <tr>
                    <td class="mono">${escapeHtml(row.file_path || "")}</td>
                    <td>${escapeHtml(row.kind || "")}</td>
                    <td>${escapeHtml(String(row.updated_at || "").slice(0, 19).replace("T", " "))}</td>
                </tr>
            `).join("");
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function runFileForgeIndex() {
    const params = new URLSearchParams({
        path: document.getElementById("file-forge-path")?.value || "eidosian_forge/doc_forge/runtime",
        max_files: document.getElementById("file-forge-max-files")?.value || "0",
        remove_after_ingest: document.getElementById("file-forge-remove")?.value || "false",
        background: "true",
    });
    await fetch(`/api/file-forge/index?${params.toString()}`, {method: "POST"});
    await refreshFileForge();
refreshWordForgeBridge();
    await refreshRuntimeServices();
}

async function refreshDocStatus() {
    try {
        const res = await fetch("/api/doc/status", {cache: "no-store"});
        if (!res.ok) return;
        const data = await res.json();
        const status = data.status || {};
        document.getElementById("live-model").textContent = status.model || "n/a";
        document.getElementById("live-processed").textContent = status.processed ?? 0;
        document.getElementById("live-remaining").textContent = status.remaining ?? 0;
        const score = Number(status.average_quality_score ?? 0);
        document.getElementById("live-score").textContent = score.toFixed(3);
        const avgSec = Number(status.average_seconds_per_document ?? 0);
        document.getElementById("live-avg-sec").textContent = avgSec.toFixed(2);
        document.getElementById("live-eta").textContent = status.eta_seconds ?? "n/a";
        document.getElementById("live-approved").textContent = status.last_approved || "n/a";
        document.getElementById("live-indexed").textContent = data.index_count ?? 0;
    } catch (err) {}
}

async function refreshConsciousness() {
    try {
        const res = await fetch("/api/consciousness", {cache: "no-store"});
        if (!res.ok) return;
        const data = await res.json();
        document.getElementById("cog-beats").textContent = data.beat_count ?? 0;
        document.getElementById("cog-errors").textContent = data.watchdog?.total_errors ?? 0;
        // If we had unity index in the health summary, we'd update it here.
    } catch (err) {}
}
setInterval(refreshDocStatus, 5000);
setInterval(refreshConsciousness, 5000);

async function refreshServices() {
    try {
        const params = new URLSearchParams();
        const service = selectedServiceTarget();
        if (service) params.set("service", service);
        const url = params.size ? `/api/services?${params.toString()}` : "/api/services";
        const res = await fetch(url, {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("service-table-body");
        if (!body) return;
        body.innerHTML = "";
        for (const row of (payload.services || [])) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${row.name}</td><td>${row.state}</td>`;
            body.appendChild(tr);
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshRuntimeServices() {
    try {
        const res = await fetch("/api/runtime/services", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("runtime-services-body");
        if (!body) return;
        body.innerHTML = "";
        for (const row of (payload.entries || [])) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${escapeHtml(row.service || "")}</td><td>${escapeHtml(row.status || "unknown")}</td><td>${escapeHtml(row.phase || "")}</td>`;
            body.appendChild(tr);
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function serviceAction(action) {
    try {
        const params = new URLSearchParams();
        const service = selectedServiceTarget();
        if (service) params.set("service", service);
        const url = params.size ? `/api/services/${action}?${params.toString()}` : `/api/services/${action}`;
        await fetch(url, {method: "POST"});
    } finally {
        await refreshServices();
        await refreshRuntimeServices();
    }
}

async function refreshArchiveState() {
    try {
        const [planRes, lifecycleRes] = await Promise.all([
            fetch('/api/code-forge/archive-plan', {cache: 'no-store'}),
            fetch('/api/code-forge/archive-lifecycle', {cache: 'no-store'}),
        ]);
        if (planRes.ok) {
            const payload = await planRes.json();
            const status = payload.status || {};
            const report = payload.report || {};
            document.getElementById('archive-plan-status').textContent = status.status || 'idle';
            document.getElementById('archive-plan-files').textContent = status.archive_files_total ?? report.archive_files_total ?? 0;
            document.getElementById('archive-plan-batches').textContent = status.batch_count ?? report.batch_count ?? 0;
        }
        if (lifecycleRes.ok) {
            const payload = await lifecycleRes.json();
            const status = payload.status || {};
            const report = payload.report || {};
            const summary = report.summary || {};
            const retirements = payload.retirements || {};
            document.getElementById('archive-lifecycle-status').textContent = status.status || 'idle';
            document.getElementById('archive-lifecycle-phase').textContent = status.phase || 'idle';
            document.getElementById('archive-lifecycle-repos').textContent = report.repo_count ?? status.repo_count ?? 0;
            document.getElementById('archive-lifecycle-ready').textContent = `${summary.retirement_ready ?? 0} / ${summary.retired ?? 0}`;
            document.getElementById('archive-lifecycle-latest-retirement').textContent = retirements.generated_at || 'n/a';
            const body = document.getElementById('archive-lifecycle-history-body');
            if (body) {
                body.innerHTML = (payload.history || []).slice(0, 8).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.status || 'idle')}</td>
                        <td>${escapeHtml(row.phase || '')}</td>
                        <td>${escapeHtml(((row.selected_repo_keys || row.repo_keys || [])).join(', '))}</td>
                        <td>${escapeHtml(String(row.finished_at || row.started_at || '').slice(0, 19).replace('T', ' '))}</td>
                    </tr>
                `).join('');
            }
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function triggerArchivePlan(refresh) {
    try {
        await fetch(`/api/code-forge/archive-plan?background=true&refresh=${refresh ? 'true' : 'false'}`, {method: 'POST'});
    } finally {
        await refreshArchiveState();
        await refreshRuntimeServices();
    }
}

async function triggerArchiveLifecycleStatus(refresh) {
    try {
        await fetch(`/api/code-forge/archive-lifecycle/status?background=true&refresh=${refresh ? 'true' : 'false'}`, {method: 'POST'});
    } finally {
        await refreshArchiveState();
        await refreshRuntimeServices();
    }
}

async function triggerArchiveWave(repoKey, batchLimit, refresh, retryFailed) {
    try {
        const params = new URLSearchParams();
        params.set('background', 'true');
        params.set('repo_key', repoKey || '');
        params.set('batch_limit', String(batchLimit || 5));
        params.set('refresh', refresh ? 'true' : 'false');
        params.set('retry_failed', retryFailed ? 'true' : 'false');
        await fetch(`/api/code-forge/archive-lifecycle/run-wave?${params.toString()}`, {method: 'POST'});
    } finally {
        await refreshArchiveState();
        await refreshRuntimeServices();
    }
}

async function previewArchiveRetire(repoKey, assumeRemoveMode) {
    try {
        const params = new URLSearchParams();
        params.set('background', 'true');
        params.set('repo_key', repoKey || '');
        params.set('assume_remove_mode', assumeRemoveMode ? 'true' : 'false');
        await fetch(`/api/code-forge/archive-lifecycle/preview-retire?${params.toString()}`, {method: 'POST'});
    } finally {
        await refreshArchiveState();
        await refreshRuntimeServices();
    }
}

async function retireArchiveRepo(repoKey, dryRun) {
    try {
        const params = new URLSearchParams();
        params.set('background', 'true');
        params.set('repo_key', repoKey || '');
        params.set('dry_run', dryRun ? 'true' : 'false');
        await fetch(`/api/code-forge/archive-lifecycle/retire?${params.toString()}`, {method: 'POST'});
    } finally {
        await refreshArchiveState();
        await refreshRuntimeServices();
    }
}

async function setArchiveRepoModePrompt() {
    const repoKey = window.prompt('Repo key to change mode for:', 'eidos_v1_concept');
    if (!repoKey) return;
    const mode = window.prompt('Mode: ingest_and_keep or ingest_and_remove', 'ingest_and_keep');
    if (!mode) return;
    const reason = window.prompt('Reason for mode change:', 'operator update') || '';
    const params = new URLSearchParams({repo_key: repoKey, mode, reason});
    await fetch(`/api/code-forge/archive-lifecycle/set-mode?${params.toString()}`, {method: 'POST'});
    await refreshArchiveState();
}

async function pruneRetiredRepo(repoKey, dryRun) {
    const params = new URLSearchParams({repo_key: repoKey, background: 'true'});
    if (dryRun) {
        params.set('dry_run', 'true');
    }
    await fetch(`/api/code-forge/archive-lifecycle/prune-retired?${params.toString()}`, {method: 'POST'});
    await refreshArchiveState();
    await refreshRuntimeServices();
}

async function restoreArchiveRepoPrompt() {
    const repoKey = window.prompt('Repo key to restore:', 'eidos_v1_concept');
    if (!repoKey) return;
    const params = new URLSearchParams({repo_key: repoKey, background: 'true'});
    await fetch(`/api/code-forge/archive-lifecycle/restore?${params.toString()}`, {method: 'POST'});
    await refreshArchiveState();
    await refreshRuntimeServices();
}

async function refreshDocs() {
    try {
        const res = await fetch("/api/docs/refresh", {method: "POST"});
        if (!res.ok) return;
        const payload = await res.json();
        const status = payload.status || payload;
        document.getElementById("docs-coverage-ratio").textContent = `${((Number(status.coverage_ratio || 0)) * 100).toFixed(1)}%`;
        document.getElementById("docs-missing-count").textContent = status.missing_readme_count ?? 0;
        document.getElementById("docs-required-count").textContent = status.required_directory_count ?? 0;
        document.getElementById("docs-review-count").textContent = status.review_pending_count ?? 0;
        document.getElementById("docs-suppressed-count").textContent = status.suppressed_directory_count ?? 0;
        document.getElementById("docs-output").textContent = (status.missing_examples || []).join("\n");
        await refreshDocsTree();
        await refreshDocsHistory();
    } catch (err) {
        // ignore transient refresh failures
    }
}

function docsPath() {
    return document.getElementById("docs-path-input").value.trim();
}

function docsBatchLimit() {
    return Math.max(1, Number(document.getElementById("docs-batch-limit").value || 20));
}

function docsMissingOnly() {
    return Boolean(document.getElementById("docs-missing-only").checked);
}

function setDocsPath(path) {
    document.getElementById("docs-path-input").value = path;
}

async function renderDocsReadme() {
    const path = docsPath();
    if (!path) return;
    const res = await fetch(`/api/docs/render?path=${encodeURIComponent(path)}`);
    const payload = await res.json();
    document.getElementById("docs-output").textContent = payload.content || JSON.stringify(payload, null, 2);
}

async function readDocsReadme() {
    const path = docsPath();
    if (!path) return;
    const res = await fetch(`/api/docs/readme?path=${encodeURIComponent(path)}`);
    const payload = await res.json();
    document.getElementById("docs-output").textContent = payload.content || JSON.stringify(payload, null, 2);
}

async function diffDocsReadme() {
    const path = docsPath();
    if (!path) return;
    const res = await fetch(`/api/docs/diff?path=${encodeURIComponent(path)}`);
    const payload = await res.json();
    document.getElementById("docs-output").textContent = payload.diff || JSON.stringify(payload, null, 2);
}

async function upsertDocsReadme() {
    const path = docsPath();
    if (!path) return;
    const res = await fetch(`/api/docs/upsert?path=${encodeURIComponent(path)}`, {method: "POST"});
    const payload = await res.json();
    document.getElementById("docs-output").textContent = JSON.stringify(payload, null, 2);
    await refreshDocs();
}

async function batchDocs(dryRun) {
    const params = new URLSearchParams({
        limit: String(docsBatchLimit()),
        missing_only: String(docsMissingOnly()),
        dry_run: String(Boolean(dryRun)),
        background: "true",
    });
    const path = docsPath();
    if (path) params.set("path_prefix", path);
    const res = await fetch(`/api/docs/upsert-batch?${params.toString()}`, {method: "POST"});
    const payload = await res.json();
    document.getElementById("docs-output").textContent = JSON.stringify(payload, null, 2);
    await refreshDocsBatchStatus();
}

async function refreshDocsTree() {
    try {
        const res = await fetch("/api/docs/tree?limit=24&refresh=true", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("docs-tree-body");
        body.innerHTML = "";
        for (const row of (payload.nodes || []).slice(0, 24)) {
            const tr = document.createElement("tr");
            const safePath = escapeHtml(row.path);
            tr.innerHTML = `<td><button onclick="setDocsPath(${JSON.stringify(row.path)})">${safePath}</button></td><td>${row.has_readme}</td><td>${row.tracked_files}</td><td>${row.tests_present}</td>`;
            body.appendChild(tr);
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshDocsHistory() {
    try {
        const res = await fetch("/api/docs/history?limit=8", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("docs-history-body");
        body.innerHTML = "";
        for (const row of (payload.entries || [])) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${(Number(row.coverage_ratio || 0) * 100).toFixed(1)}%</td><td>${row.missing_readme_count ?? 0}</td><td>${row.missing_delta ?? 0}</td><td>${row.drift_state || 'stable'}</td>`;
            body.appendChild(tr);
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshDocsBatchStatus() {
    try {
        const res = await fetch("/api/docs/upsert-batch/status", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const line = document.getElementById("docs-batch-status-line");
        if (!line) return;
        const status = typeof payload.status === "object" && payload.status !== null ? payload.status : payload;
        const summary = [
            `batch: ${status.state || 'idle'}`,
            `dry_run=${Boolean(status.dry_run)}`,
            `processed=${status.processed ?? 0}`,
            `changed=${status.changed ?? status.changed_count ?? status.result?.changed_count ?? 0}`,
            `failed=${status.failed ?? 0}`,
        ];
        if (status.path_prefix) summary.push(`path=${status.path_prefix}`);
        if (status.updated_at || status.finished_at) summary.push(`updated=${status.updated_at || status.finished_at}`);
        line.textContent = summary.join(" · ");
        if ((status.state || status.status || "") === "completed") {
            await refreshDocs();
        }
        const batchStatus = document.getElementById("docs-batch-status");
        if (batchStatus) batchStatus.textContent = String(status.status || status.state || "idle").toUpperCase();
        const batchLimit = document.getElementById("docs-batch-limit-view");
        if (batchLimit) batchLimit.textContent = status.limit ?? "";
        const batchPath = document.getElementById("docs-batch-path");
        if (batchPath) batchPath.textContent = status.path_prefix || "forge";
        const batchDryRun = document.getElementById("docs-batch-dry-run");
        if (batchDryRun) batchDryRun.textContent = String(Boolean(status.dry_run));
        const batchFinished = document.getElementById("docs-batch-finished");
        if (batchFinished) batchFinished.textContent = String(status.finished_at || "").slice(0, 19).replace("T", " ");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshDocsBatchHistory() {
    try {
        const res = await fetch("/api/docs/upsert-batch/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("docs-batch-history-body");
        if (!body) return;
        body.innerHTML = (payload.entries || []).slice(0, 6).map((row) => `
            <tr>
                <td>${escapeHtml(row.status || "unknown")}</td>
                <td>${escapeHtml(row.limit ?? "")}</td>
                <td>${escapeHtml(row.path_prefix || "forge")}</td>
                <td>${escapeHtml(String(row.finished_at || "").slice(0, 19).replace("T", " "))}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshSessionBridge() {
    try {
        const res = await fetch("/api/session-bridge", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const summary = payload.summary || {};
        const geminiImported = summary.gemini_records ?? 0;
        const codexImported = summary.codex_records ?? 0;
        document.getElementById("session-bridge-count").textContent = (payload.recent_sessions || []).length;
        document.getElementById("session-bridge-imported").textContent = geminiImported + codexImported;
        document.getElementById("session-bridge-sync").textContent = summary.last_sync_at || "n/a";
        document.getElementById("session-bridge-codex").textContent = codexImported;
        document.getElementById("session-bridge-gemini").textContent = geminiImported;
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshProofBundle() {
    try {
        const res = await fetch("/api/proof/bundle/latest", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        document.getElementById("proof-bundle-benchmarks").textContent = (payload.benchmarks || []).length;
        document.getElementById("proof-bundle-missing").textContent = (payload.missing || []).length;
        document.getElementById("proof-bundle-root").textContent = payload.bundle_root || "n/a";
        document.getElementById("proof-bundle-score").textContent = payload.proof_summary?.score ?? "n/a";
        document.getElementById("proof-bundle-status").textContent = payload.proof_summary?.status || "n/a";
        document.getElementById("proof-bundle-identity-trend").textContent = payload.identity_summary?.history?.trend || "n/a";
        document.getElementById("proof-bundle-identity-delta").textContent =
            payload.identity_summary?.history?.delta_from_previous ?? "n/a";
        document.getElementById("proof-bundle-session-imports").textContent =
            payload.session_bridge_summary?.imported_records ?? 0;
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshIdentityContinuity() {
    try {
        const res = await fetch("/api/proof/identity/latest", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        document.getElementById("identity-score").textContent = payload.overall_score ?? "n/a";
        document.getElementById("identity-status").textContent = payload.status || "missing";
        document.getElementById("identity-sessions").textContent =
            payload.session_bridge?.recent_sessions ?? 0;
        document.getElementById("identity-trend").textContent =
            payload.history?.trend || "n/a";
        document.getElementById("identity-delta").textContent =
            payload.history?.delta_from_previous ?? "n/a";
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshIdentityHistory() {
    try {
        const res = await fetch("/api/proof/identity/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const entries = payload.entries || [];
        document.getElementById("identity-history-count").textContent = entries.length;
        const body = document.getElementById("identity-history-body");
        body.innerHTML = entries.slice(-6).map((row) => `
            <tr>
                <td>${row.generated_at || ""}</td>
                <td>${row.status || ""}</td>
                <td>${row.overall_score ?? ""}</td>
                <td>${row.recent_sessions ?? 0}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshProofHistory() {
    try {
        const res = await fetch("/api/proof/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const entries = payload.entries || [];
        const body = document.getElementById("proof-history-body");
        body.innerHTML = entries.slice(-6).map((row) => `
            <tr>
                <td>${row.generated_at || ""}</td>
                <td>${row.status || ""}</td>
                <td>${row.overall_score ?? ""}</td>
                <td>${row.freshness_status || ""}</td>
                <td>${row.regression_status || ""}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshExternalBenchmarks() {
    try {
        const res = await fetch("/api/proof/external", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const entries = payload.entries || [];
        const body = document.getElementById("external-benchmarks-body");
        body.innerHTML = entries.slice(0, 8).map((row) => `
            <tr>
                <td>${row.suite || ""}</td>
                <td>${row.execution_mode || ""}</td>
                <td>${row.status || ""}</td>
                <td>${row.score ?? ""}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshRuntimeBenchmarks() {
    try {
        const res = await fetch("/api/benchmarks/runtime", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const entries = payload.entries || [];
        const body = document.getElementById("runtime-benchmarks-body");
        body.innerHTML = entries.slice(0, 8).map((row) => `
            <tr>
                <td>${row.scenario || ""}</td>
                <td>${row.engine || ""}</td>
                <td>${row.status || ""}</td>
                <td>${row.completed_count ?? 0}</td>
                <td>${row.attempt_count ?? 0}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshProofRefreshStatus() {
    try {
        const res = await fetch("/api/proof/refresh/status", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        document.getElementById("proof-refresh-status").textContent = String(payload.status || "idle").toUpperCase();
        document.getElementById("proof-refresh-window").textContent = payload.window_days ?? 30;
        document.getElementById("proof-refresh-started").textContent = (payload.started_at || "").slice(0, 19).replace("T", " ");
        document.getElementById("proof-refresh-proof-rc").textContent = payload.proof_returncode ?? "";
        document.getElementById("proof-refresh-bundle-rc").textContent = payload.bundle_returncode ?? "";
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshProofRefreshHistory() {
    try {
        const res = await fetch("/api/proof/refresh/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("proof-refresh-history-body");
        if (!body) return;
        body.innerHTML = (payload.entries || []).slice(0, 6).map((row) => `
            <tr>
                <td>${escapeHtml(row.status || "unknown")}</td>
                <td>${escapeHtml(row.window_days ?? "")}</td>
                <td>${escapeHtml(row.proof_returncode ?? "")}</td>
                <td>${escapeHtml(String(row.finished_at || "").slice(0, 19).replace("T", " "))}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function triggerProofRefresh() {
    try {
        await fetch("/api/proof/refresh?background=true&window_days=30", {method: "POST"});
    } finally {
        await refreshProofRefreshStatus();
        await refreshProofRefreshHistory();
    }
}

async function refreshRuntimeBenchmarkRunStatus() {
    try {
        const res = await fetch("/api/benchmarks/runtime/run/status", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        document.getElementById("benchmark-run-status").textContent = String(payload.status || "idle").toUpperCase();
        document.getElementById("benchmark-run-scenario").textContent = payload.scenario || "scenario2";
        document.getElementById("benchmark-run-engine").textContent = payload.engine || "local_agent";
        document.getElementById("benchmark-run-returncode").textContent = payload.returncode ?? "";
        document.getElementById("benchmark-run-finished").textContent = (payload.finished_at || "").slice(0, 19).replace("T", " ");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshRuntimeBenchmarkRunHistory() {
    try {
        const res = await fetch("/api/benchmarks/runtime/run/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("benchmark-run-history-body");
        if (!body) return;
        body.innerHTML = (payload.entries || []).slice(0, 6).map((row) => `
            <tr>
                <td>${escapeHtml(row.status || "unknown")}</td>
                <td>${escapeHtml(row.scenario || "")}</td>
                <td>${escapeHtml(row.engine || "")}</td>
                <td>${escapeHtml(String(row.finished_at || "").slice(0, 19).replace("T", " "))}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function triggerRuntimeBenchmark(engine) {
    try {
        await fetch(`/api/benchmarks/runtime/run?background=true&scenario=scenario2&engine=${encodeURIComponent(engine || "local_agent")}&attempts_per_step=1&timeout_sec=900&keep_alive=4h`, {method: "POST"});
    } finally {
        await refreshRuntimeBenchmarkRunStatus();
        await refreshRuntimeBenchmarkRunHistory();
    }
}

async function refreshRuntimeArtifactAuditStatus() {
    try {
        const res = await fetch("/api/runtime-artifacts/audit/status", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        document.getElementById("runtime-audit-status").textContent = String(payload.status || "idle").toUpperCase();
        document.getElementById("runtime-audit-tracked").textContent = payload.tracked_violation_count ?? "";
        document.getElementById("runtime-audit-live").textContent = payload.live_generated_count ?? "";
        document.getElementById("runtime-audit-report").textContent = payload.latest_report || "n/a";
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshRuntimeArtifactAuditHistory() {
    try {
        const res = await fetch("/api/runtime-artifacts/audit/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("runtime-audit-history-body");
        if (!body) return;
        body.innerHTML = (payload.entries || []).slice(0, 6).map((row) => `
            <tr>
                <td>${escapeHtml(row.status || "unknown")}</td>
                <td>${escapeHtml(row.tracked_violation_count ?? "")}</td>
                <td>${escapeHtml(row.live_generated_count ?? "")}</td>
                <td>${escapeHtml(String(row.finished_at || "").slice(0, 19).replace("T", " "))}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function triggerRuntimeArtifactAudit() {
    try {
        await fetch("/api/runtime-artifacts/audit?background=true", {method: "POST"});
    } finally {
        await refreshRuntimeArtifactAuditStatus();
        await refreshRuntimeArtifactAuditHistory();
    }
}

async function refreshCodeForgeProvenanceAuditStatus() {
    try {
        const res = await fetch("/api/code-forge/provenance-audit/status", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        document.getElementById("provenance-audit-status").textContent = String(payload.status || "idle").toUpperCase();
        document.getElementById("provenance-audit-links").textContent = payload.link_file_count ?? "";
        document.getElementById("provenance-audit-registries").textContent = payload.registry_file_count ?? "";
        document.getElementById("provenance-audit-invalid").textContent = payload.invalid_file_count ?? "";
        document.getElementById("provenance-audit-report").textContent = payload.latest_report || "n/a";
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function refreshCodeForgeProvenanceAuditHistory() {
    try {
        const res = await fetch("/api/code-forge/provenance-audit/history", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const body = document.getElementById("provenance-audit-history-body");
        if (!body) return;
        body.innerHTML = (payload.entries || []).slice(0, 6).map((row) => `
            <tr>
                <td>${escapeHtml(row.status || "unknown")}</td>
                <td>${escapeHtml(row.link_file_count ?? "")}</td>
                <td>${escapeHtml(row.registry_file_count ?? "")}</td>
                <td>${escapeHtml(String(row.finished_at || "").slice(0, 19).replace("T", " "))}</td>
            </tr>
        `).join("");
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function triggerCodeForgeProvenanceAudit() {
    try {
        await fetch("/api/code-forge/provenance-audit?background=true", {method: "POST"});
    } finally {
        await refreshCodeForgeProvenanceAuditStatus();
        await refreshCodeForgeProvenanceAuditHistory();
    }
}

async function syncSessionBridge() {
    try {
        const res = await fetch("/api/session-bridge/sync", {method: "POST"});
        const payload = await res.json();
        document.getElementById("docs-output").textContent = JSON.stringify(payload, null, 2);
    } finally {
        await refreshSessionBridge();
    }
}

async function refreshScheduler() {
    try {
        const res = await fetch("/api/scheduler", {cache: "no-store"});
        if (!res.ok) return;
        const payload = await res.json();
        const state = payload.payload?.state || {};
        const status = payload.payload?.status || {};
        document.getElementById("scheduler-state").textContent = status.state || "unknown";
        document.getElementById("scheduler-cycle").textContent = status.cycle ?? state.cycle ?? 0;
        document.getElementById("scheduler-task").textContent = status.current_task || "idle";
        document.getElementById("scheduler-next-run").textContent = status.next_run_in_seconds ?? 0;
        document.getElementById("scheduler-pause-flag").textContent = String(Boolean(state.pause_requested));
        document.getElementById("scheduler-stop-flag").textContent = String(Boolean(state.stop_requested));
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function schedulerAction(action) {
    try {
        await fetch(`/api/scheduler/${action}`, {method: "POST"});
    } finally {
        await refreshScheduler();
    }
}

setInterval(refreshServices, 8000);
setInterval(refreshScheduler, 8000);
setInterval(refreshDocsHistory, 12000);
setInterval(refreshDocsBatchStatus, 5000);
setInterval(refreshDocsBatchHistory, 12000);
setInterval(refreshSessionBridge, 15000);
setInterval(refreshProofBundle, 15000);
setInterval(refreshIdentityContinuity, 15000);
setInterval(refreshIdentityHistory, 15000);
setInterval(refreshProofHistory, 15000);
setInterval(refreshExternalBenchmarks, 15000);
setInterval(refreshRuntimeBenchmarks, 15000);
setInterval(refreshProofRefreshStatus, 10000);
setInterval(refreshProofRefreshHistory, 15000);
setInterval(refreshRuntimeBenchmarkRunStatus, 10000);
setInterval(refreshRuntimeBenchmarkRunHistory, 15000);
setInterval(refreshRuntimeArtifactAuditStatus, 10000);
setInterval(refreshRuntimeArtifactAuditHistory, 15000);
setInterval(refreshCodeForgeProvenanceAuditStatus, 10000);
setInterval(refreshCodeForgeProvenanceAuditHistory, 15000);
setInterval(refreshRuntimeServices, 10000);
setInterval(refreshArchiveState, 12000);
setInterval(refreshFileForge, 15000);
setInterval(refreshWordForgeBridge, 15000);
refreshScheduler();
refreshServices();
refreshRuntimeServices();
refreshArchiveState();
refreshFileForge();
refreshWordForgeBridge();
refreshDocsTree();
refreshDocsHistory();
refreshDocsBatchStatus();
refreshDocsBatchHistory();
refreshSessionBridge();
refreshProofBundle();
refreshIdentityContinuity();
refreshIdentityHistory();
refreshProofHistory();
refreshExternalBenchmarks();
refreshRuntimeBenchmarks();
refreshProofRefreshStatus();
refreshProofRefreshHistory();
refreshRuntimeBenchmarkRunStatus();
refreshRuntimeBenchmarkRunHistory();
refreshRuntimeArtifactAuditStatus();
refreshRuntimeArtifactAuditHistory();
refreshCodeForgeProvenanceAuditStatus();
refreshCodeForgeProvenanceAuditHistory();

async function refreshWordForgeBridge() {
    try {
        const [multiRes, fasttextRes, polyglotRes, bridgeRes, multiHistRes, fasttextHistRes, polyglotHistRes, bridgeHistRes] = await Promise.all([
            fetch('/api/word-forge/multilingual', {cache: 'no-store'}),
            fetch('/api/word-forge/fasttext', {cache: 'no-store'}),
            fetch('/api/word-forge/polyglot', {cache: 'no-store'}),
            fetch('/api/word-forge/bridge-audit', {cache: 'no-store'}),
            fetch('/api/word-forge/multilingual/history', {cache: 'no-store'}),
            fetch('/api/word-forge/fasttext/history', {cache: 'no-store'}),
            fetch('/api/word-forge/polyglot/history', {cache: 'no-store'}),
            fetch('/api/word-forge/bridge-audit/history', {cache: 'no-store'}),
        ]);
        if (multiRes.ok) {
            const payload = await multiRes.json();
            const status = payload.status || {};
            const report = payload.latest_report || {};
            document.getElementById('wf-multi-status').textContent = status.status || 'idle';
            document.getElementById('wf-multi-phase').textContent = status.phase || 'idle';
            document.getElementById('wf-multi-lexemes').textContent = report.after?.lexeme_count ?? 0;
            document.getElementById('wf-multi-translations').textContent = report.after?.translation_count ?? 0;
            document.getElementById('wf-multi-base').textContent = report.after?.base_aligned_count ?? 0;
        }
        if (fasttextRes.ok) {
            const payload = await fasttextRes.json();
            const status = payload.status || {};
            const report = payload.latest_report || {};
            const after = report.after?.fasttext || {};
            document.getElementById('wf-fasttext-status').textContent = status.status || 'idle';
            document.getElementById('wf-fasttext-phase').textContent = status.phase || 'idle';
            document.getElementById('wf-fasttext-vectors').textContent = after.vector_count ?? 0;
            document.getElementById('wf-fasttext-candidates').textContent = after.candidate_count ?? 0;
            document.getElementById('wf-fasttext-applied').textContent = after.applied_count ?? 0;
            document.getElementById('wf-fasttext-languages').textContent = after.language_count ?? 0;
        }
        if (polyglotRes.ok) {
            const payload = await polyglotRes.json();
            const status = payload.status || {};
            const report = payload.latest_report || {};
            const after = report.after || {};
            document.getElementById('wf-polyglot-status').textContent = status.status || 'idle';
            document.getElementById('wf-polyglot-phase').textContent = status.phase || 'idle';
            document.getElementById('wf-polyglot-processed').textContent = report.processed_lexemes ?? 0;
            document.getElementById('wf-polyglot-multipart').textContent = report.multi_part_lexemes ?? 0;
            document.getElementById('wf-polyglot-decomposed').textContent = after.decomposed_lexeme_count ?? 0;
            document.getElementById('wf-polyglot-rows').textContent = after.lexeme_morpheme_count ?? 0;
        }
        if (bridgeRes.ok) {
            const payload = await bridgeRes.json();
            const status = payload.status || {};
            const report = payload.latest_report || {};
            const counts = report.bridge_counts || {};
            const quality = payload.bridge_quality || report.bridge_quality || {};
            const communities = payload.community_summary || {};
            const summary = payload || {};
            document.getElementById('wf-bridge-status').textContent = status.status || 'idle';
            document.getElementById('wf-bridge-phase').textContent = status.phase || 'idle';
            document.getElementById('wf-bridge-candidates').textContent = quality.candidate_term_count ?? 0;
            document.getElementById('wf-bridge-word').textContent = counts.word ?? 0;
            document.getElementById('wf-bridge-knowledge').textContent = counts.knowledge ?? 0;
            document.getElementById('wf-bridge-code').textContent = counts.code ?? 0;
            document.getElementById('wf-bridge-file').textContent = counts.file ?? 0;
            document.getElementById('wf-bridge-morpheme').textContent = counts.morpheme ?? 0;
            document.getElementById('wf-bridge-morph-linked').textContent = counts.morphologically_linked ?? 0;
            document.getElementById('wf-bridge-full').textContent = counts.fully_bridged ?? 0;
            document.getElementById('wf-bridge-delta').textContent = summary.history_summary?.delta_fully_bridged ?? 0;
            document.getElementById('wf-bridge-partial').textContent = counts.partially_bridged ?? 0;
            document.getElementById('wf-bridge-any').textContent = counts.any_bridged ?? 0;
            document.getElementById('wf-bridge-ratio').textContent = quality.fully_bridged_ratio ?? 0;
            document.getElementById('wf-bridge-communities').textContent = communities.community_count ?? 0;
            document.getElementById('wf-bridge-morpheme-count').textContent = summary.morpheme_metrics?.morpheme_count ?? 0;
            document.getElementById('wf-bridge-decomposed').textContent = summary.morpheme_metrics?.decomposed_lexeme_count ?? 0;
            document.getElementById('wf-bridge-code-units').textContent = summary.code_metrics?.code_library_unit_count ?? 0;
            document.getElementById('wf-bridge-file-links').textContent = summary.file_metrics?.link_count ?? 0;
            const body = document.getElementById('wf-bridge-history-body');
            if (body) {
                body.innerHTML = (report.top_bridged_terms || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.term || '')}</td>
                        <td>${row.word_match ? 1 : 0}</td>
                        <td>${row.knowledge_match ? 1 : 0}</td>
                        <td>${row.code_match ? 1 : 0}</td>
                        <td>${row.file_match ? 1 : 0}</td>
                        <td>${escapeHtml((row.morphemes || []).slice(0, 4).join(', '))}</td>
                    </tr>
                `).join('');
            }
            const morphemeBody = document.getElementById('wf-bridge-morpheme-body');
            if (morphemeBody) {
                morphemeBody.innerHTML = (summary.morpheme_metrics?.recent_decompositions || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.term || '')}</td>
                        <td>${escapeHtml(row.lang || '')}</td>
                        <td>${escapeHtml(row.source_type || '')}</td>
                        <td>${escapeHtml((row.morphemes || []).join(', '))}</td>
                    </tr>
                `).join('');
            }
            const communityBody = document.getElementById('wf-bridge-community-body');
            if (communityBody) {
                communityBody.innerHTML = (communities.top_communities || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.anchor_term || '')}</td>
                        <td>${escapeHtml(String(row.layer_count ?? 0))}</td>
                        <td>${escapeHtml(String(row.neighbor_count ?? 0))}</td>
                        <td>${escapeHtml(String(row.morpheme_nodes ?? 0))}</td>
                        <td>${escapeHtml((row.layers || []).join(', '))}</td>
                    </tr>
                `).join('');
            }
        }
        if (multiHistRes.ok) {
            const payload = await multiHistRes.json();
            const body = document.getElementById('wf-multi-history-body');
            if (body) {
                body.innerHTML = (payload.entries || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.status || '')}</td>
                        <td>${escapeHtml(row.phase || '')}</td>
                        <td>${escapeHtml(row.lexeme_delta ?? 0)}</td>
                        <td>${escapeHtml(String(row.finished_at || row.started_at || '').slice(0, 19).replace('T', ' '))}</td>
                    </tr>
                `).join('');
            }
        }
        if (fasttextHistRes.ok) {
            const payload = await fasttextHistRes.json();
            const body = document.getElementById('wf-fasttext-history-body');
            if (body) {
                body.innerHTML = (payload.entries || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.status || '')}</td>
                        <td>${escapeHtml(row.phase || '')}</td>
                        <td>${escapeHtml(row.vector_delta ?? 0)}</td>
                        <td>${escapeHtml(String(row.finished_at || row.started_at || '').slice(0, 19).replace('T', ' '))}</td>
                    </tr>
                `).join('');
            }
        }
        if (polyglotHistRes.ok) {
            const payload = await polyglotHistRes.json();
            const body = document.getElementById('wf-polyglot-history-body');
            if (body) {
                body.innerHTML = (payload.entries || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.status || '')}</td>
                        <td>${escapeHtml(row.phase || '')}</td>
                        <td>${escapeHtml(row.decomposed_lexeme_delta ?? 0)}</td>
                        <td>${escapeHtml(String(row.finished_at || row.started_at || '').slice(0, 19).replace('T', ' '))}</td>
                    </tr>
                `).join('');
            }
        }
        if (bridgeHistRes.ok) {
            const payload = await bridgeHistRes.json();
            const body = document.getElementById('wf-bridge-run-history-body');
            if (body) {
                body.innerHTML = (payload.entries || []).slice(0, 10).map((row) => `
                    <tr>
                        <td>${escapeHtml(row.status || '')}</td>
                        <td>${escapeHtml(row.phase || '')}</td>
                        <td>${escapeHtml((row.status || '') === 'completed' ? 'yes' : 'no')}</td>
                        <td>${escapeHtml(String(row.finished_at || row.started_at || '').slice(0, 19).replace('T', ' '))}</td>
                    </tr>
                `).join('');
            }
        }
    } catch (err) {
        // ignore transient refresh failures
    }
}

async function runWordForgePolyglot() {
    const lang = document.getElementById('wf-polyglot-lang')?.value?.trim();
    const limitValue = document.getElementById('wf-polyglot-limit')?.value?.trim();
    const force = Boolean(document.getElementById('wf-polyglot-force')?.checked);
    const params = new URLSearchParams();
    if (lang) params.set('lang', lang);
    if (limitValue) params.set('limit', limitValue);
    if (force) params.set('force', 'true');
    try {
        await fetch(`/api/word-forge/polyglot/run?${params.toString()}`, {method: 'POST'});
    } finally {
        await refreshWordForgeBridge();
        await refreshRuntimeServices();
    }
}

async function runWordForgeFastTextIngest() {
    const sourcePath = document.getElementById('wf-fasttext-source-path')?.value?.trim();
    const lang = document.getElementById('wf-fasttext-lang')?.value?.trim();
    const bootstrapLang = document.getElementById('wf-fasttext-bootstrap-lang')?.value?.trim();
    const limitValue = document.getElementById('wf-fasttext-limit')?.value?.trim();
    const minScore = document.getElementById('wf-fasttext-min-score')?.value?.trim();
    const apply = Boolean(document.getElementById('wf-fasttext-apply')?.checked);
    const force = Boolean(document.getElementById('wf-fasttext-force')?.checked);
    if (!sourcePath || !lang) return;
    const params = new URLSearchParams({source_path: sourcePath, lang});
    if (bootstrapLang) params.set('bootstrap_lang', bootstrapLang);
    if (limitValue) params.set('limit', limitValue);
    if (minScore) params.set('min_score', minScore);
    if (apply) params.set('apply', 'true');
    if (force) params.set('force', 'true');
    try {
        await fetch(`/api/word-forge/fasttext/run?${params.toString()}`, {method: 'POST'});
    } finally {
        await refreshWordForgeBridge();
        await refreshRuntimeServices();
    }
}

async function runWordForgeMultilingualIngest() {
    const sourcePath = document.getElementById('wf-source-path')?.value?.trim();
    const sourceType = document.getElementById('wf-source-type')?.value || 'wiktextract';
    const limitValue = document.getElementById('wf-source-limit')?.value?.trim();
    const force = Boolean(document.getElementById('wf-source-force')?.checked);
    if (!sourcePath) return;
    const params = new URLSearchParams({source_path: sourcePath, source_type: sourceType});
    if (limitValue) params.set('limit', limitValue);
    if (force) params.set('force', 'true');
    try {
        await fetch(`/api/word-forge/multilingual/run?${params.toString()}`, {method: 'POST'});
    } finally {
        await refreshWordForgeBridge();
        await refreshRuntimeServices();
    }
}

async function runWordForgeBridgeAudit() {
    try {
        await fetch('/api/word-forge/bridge-audit/run', {method: 'POST'});
    } finally {
        await refreshWordForgeBridge();
        await refreshRuntimeServices();
    }
}
