"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
// ──────────────────────────────────────────────
// CONFIGURATION
// ──────────────────────────────────────────────
const RAG_API_URL = 'http://localhost:8321';
const HEALTH_TIMEOUT_MS = 5000; // 5 sec — fail fast if server is down
const STREAM_TIMEOUT_MS = 90000; // 90 sec to receive the FIRST token
// ──────────────────────────────────────────────
// Helper: health check
// ──────────────────────────────────────────────
async function checkHealth() {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
    try {
        const response = await fetch(`${RAG_API_URL}/health`, {
            method: 'GET',
            signal: controller.signal,
        });
        return response.ok;
    }
    catch {
        return false;
    }
    finally {
        clearTimeout(timeout);
    }
}
// ──────────────────────────────────────────────
// Helper: stream from /chat/stream and pipe tokens to VS Code
//
// TIMEOUT STRATEGY:
// - STREAM_TIMEOUT_MS (90s) guards the wait for the FIRST data event only.
// - The server sends {"status":"working"} immediately at t=0, which resolves
//   reader.read() right away — clearing firstTokenReceived and disarming the
//   AbortController before any real processing begins.
// - Once any data arrives (including the status event), we clear the timeout.
//   The stream can then run as long as needed without VS Code killing it.
//
// WHY {"status":"working"} IS NEEDED (not just SSE heartbeat comments):
// - Node.js fetch (Undici) buffers ~16KB before delivering data to reader.read()
// - SSE comment heartbeats (": heartbeat\n\n") are ~13 bytes — they fill the
//   buffer far too slowly, so reader.read() never resolves within 90s
// - A proper "data: {...}\n\n" line is flushed by Undici immediately
// ──────────────────────────────────────────────
async function streamRAG(message, resetMemory, vsStream, cancellationToken) {
    const controller = new AbortController();
    const firstTokenTimeout = setTimeout(() => controller.abort(), STREAM_TIMEOUT_MS);
    let firstTokenReceived = false;
    const response = await fetch(`${RAG_API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, reset_memory: resetMemory }),
        signal: controller.signal,
    });
    if (!response.ok) {
        clearTimeout(firstTokenTimeout);
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
    }
    if (!response.body) {
        clearTimeout(firstTokenTimeout);
        throw new Error('No response body from phonex server');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let sources = [];
    try {
        while (true) {
            if (cancellationToken.isCancellationRequested) {
                reader.cancel();
                break;
            }
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            // Clear the first-token timeout the moment ANY data arrives.
            // The server sends {"status":"working"} immediately at t=0,
            // so this fires within milliseconds of the connection opening.
            if (!firstTokenReceived) {
                firstTokenReceived = true;
                clearTimeout(firstTokenTimeout);
            }
            buffer += decoder.decode(value, { stream: true });
            // SSE events are separated by \n\n
            const events = buffer.split('\n\n');
            buffer = events.pop() ?? ''; // keep any incomplete trailing event
            for (const event of events) {
                const line = event.trim();
                // Skip SSE comment lines (heartbeats: lines starting with ':')
                // These are valid SSE but carry no data — just ignore them.
                if (line.startsWith(':')) {
                    continue;
                }
                if (!line.startsWith('data:')) {
                    continue;
                }
                const jsonStr = line.slice('data:'.length).trim();
                if (!jsonStr) {
                    continue;
                }
                let chunk;
                try {
                    chunk = JSON.parse(jsonStr);
                }
                catch {
                    continue;
                }
                // Server-side error surfaced through the stream
                if (chunk.error) {
                    vsStream.markdown(`\n\n❌ **Server error:** ${chunk.error}`);
                    return;
                }
                // Immediate flush event — server sends this at t=0 to bypass
                // Node.js/Undici buffering. Nothing to display, just skip it.
                if (chunk.status === 'working') {
                    continue;
                }
                if (chunk.token) {
                    vsStream.markdown(chunk.token);
                }
                if (chunk.done && chunk.sources) {
                    sources = chunk.sources;
                }
            }
        }
    }
    finally {
        clearTimeout(firstTokenTimeout);
        reader.releaseLock();
    }
    // Render source citations after answer is complete
    if (sources.length > 0) {
        vsStream.markdown('\n\n---\n**Sources:**');
        for (const source of sources) {
            const score = source.score !== null ? ` (relevance: ${source.score})` : '';
            vsStream.markdown(`\n- \`${source.file}\`${score}`);
        }
    }
}
// ──────────────────────────────────────────────
// Chat Participant Handler
// ──────────────────────────────────────────────
const handler = async (request, context, stream, token) => {
    // ── /reset ──
    if (request.command === 'reset') {
        try {
            await fetch(`${RAG_API_URL}/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            stream.markdown('✅ Conversation memory cleared. Ask a new question!');
        }
        catch {
            stream.markdown('⚠️ Could not reach the phonex server to reset memory.');
        }
        return {};
    }
    // ── Health check ──
    const isHealthy = await checkHealth();
    if (!isHealthy) {
        stream.markdown('❌ **phonex server is not running.**\n\n' +
            'Start it with:\n' +
            '```powershell\n' +
            'cd D:\\Gitrnd\\phonex\n' +
            '$env:OLLAMA_GPU_LAYERS = "20"\n' +
            '$env:OLLAMA_FLASH_ATTENTION = "1"\n' +
            '.\\.venv\\Scripts\\Activate.ps1\n' +
            'python server.py\n' +
            '```');
        return {};
    }
    // ── Build prompt ──
    let prompt = request.prompt;
    if (request.command === 'flow') {
        prompt = `Explain the data flow or request lifecycle for: ${request.prompt}`;
    }
    else if (request.command === 'pattern') {
        prompt = `Show the code pattern or best practice for: ${request.prompt}`;
    }
    else if (request.command === 'package') {
        prompt = `Provide details about this NuGet package: ${request.prompt}. Include its purpose, key classes, dependencies, and usage examples.`;
    }
    stream.progress('Searching codebase...');
    try {
        await streamRAG(prompt, false, stream, token);
    }
    catch (error) {
        if (error.name === 'AbortError') {
            stream.markdown('⏱️ **No response within 90 seconds.**\n\n' +
                'The model may still be loading. Check the server terminal — ' +
                'if you see Ollama activity, wait a moment and try again.\n\n' +
                'If this keeps happening, run in PowerShell:\n' +
                '```powershell\n' +
                'ollama run qwen2.5-coder:7b-instruct-q4_K_M "hello"\n' +
                '```\n' +
                'to pre-warm the model before using the extension.');
        }
        else {
            stream.markdown(`❌ **Error:** ${error.message}`);
        }
    }
    return {};
};
// ──────────────────────────────────────────────
// Extension Activation
// ──────────────────────────────────────────────
function activate(extensionContext) {
    const participant = vscode.chat.createChatParticipant('phonex.chat', handler);
    participant.iconPath = vscode.Uri.joinPath(extensionContext.extensionUri, 'icon.png');
    extensionContext.subscriptions.push(participant);
    console.log('phonex chat participant activated');
}
function deactivate() { }
//# sourceMappingURL=extension.js.map