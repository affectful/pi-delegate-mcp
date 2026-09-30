#!/usr/bin/env python3
"""MCP server that lets an agent consult or delegate to other models through Pi.

consult:  a read-only peer (default openai-codex/gpt-6-astra, high thinking) that can
          read the repository and answer; sessions can be continued for follow-ups.
delegate: a worker (default openai-codex/gpt-6.1-sol, medium thinking) with Pi's edit
          tools that changes the checkout directly; synchronous or as a background job.

Pi runs headless (`pi -p`) with skills and extensions disabled, using Pi's own login
for the model provider. The file is self-contained (standard library only) so it can
also be copied and run as a script.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVER_NAME = "pi-delegate"
SERVER_VERSION = "1.0.0"
READ_ONLY_TOOLS = "read,grep,find,ls"
THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
MAX_OUTPUT_CHARS = 60_000
MAX_PROMPT_CHARS = 200_000

CONSULT_PREAMBLE = (
    "You are being consulted by another coding agent as a peer on a hard problem. "
    "You can read the repository (read, grep, find, ls) but cannot change it. "
    "Think carefully, check claims against the code, and answer directly: say what you "
    "agree with, what you would do differently and why, and name the files and lines "
    "that matter.\n\n"
)
DELEGATE_PREAMBLE = (
    "Another coding agent delegated this task to you. Work directly in this checkout; "
    "other agents may be editing it too, so change only what the task requires and do "
    "not revert changes you did not make. Do not commit, push, or open pull requests "
    "unless the task says to. When done, reply with a concise summary of what you "
    "changed (files and why), anything you could not finish, and how you verified it.\n\n"
)


def default_state_dir() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "pi-delegate"


@dataclass
class Config:
    pi: str = "pi"
    provider: str = "openai-codex"
    consult_model: str = "gpt-6-astra"
    delegate_model: str = "gpt-6.1-sol"
    consult_thinking: str = "high"
    delegate_thinking: str = "medium"
    cwd: Path = field(default_factory=Path.cwd)
    root: Path | None = None
    extensions: list[str] = field(default_factory=list)
    state_dir: Path = field(default_factory=default_state_dir)

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "sessions"

    @property
    def job_dir(self) -> Path:
        return self.state_dir / "jobs"


def parse_config(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        prog="pi-delegate-mcp",
        description="MCP stdio server to consult or delegate to other models through Pi. "
                    "Every option can also be set with the PI_DELEGATE_* variable shown.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SERVER_VERSION}")
    parser.add_argument("--pi", default=env.get("PI_DELEGATE_PI_BIN", "pi"),
                        help="Pi executable (PI_DELEGATE_PI_BIN; default: pi on PATH)")
    parser.add_argument("--provider", default=env.get("PI_DELEGATE_PROVIDER", "openai-codex"),
                        help="provider for model ids given without one (PI_DELEGATE_PROVIDER; default: openai-codex)")
    parser.add_argument("--consult-model", default=env.get("PI_DELEGATE_CONSULT_MODEL", "gpt-6-astra"),
                        help="default consult model (PI_DELEGATE_CONSULT_MODEL; default: gpt-6-astra)")
    parser.add_argument("--delegate-model", default=env.get("PI_DELEGATE_DELEGATE_MODEL", "gpt-6.1-sol"),
                        help="default delegate model (PI_DELEGATE_DELEGATE_MODEL; default: gpt-6.1-sol)")
    parser.add_argument("--consult-thinking", default=env.get("PI_DELEGATE_CONSULT_THINKING", "high"),
                        choices=sorted(THINKING_LEVELS), help="(PI_DELEGATE_CONSULT_THINKING; default: high)")
    parser.add_argument("--delegate-thinking", default=env.get("PI_DELEGATE_DELEGATE_THINKING", "medium"),
                        choices=sorted(THINKING_LEVELS), help="(PI_DELEGATE_DELEGATE_THINKING; default: medium)")
    parser.add_argument("--cwd", default=env.get("PI_DELEGATE_CWD"),
                        help="default working directory (PI_DELEGATE_CWD; default: the server's working directory)")
    parser.add_argument("--root", default=env.get("PI_DELEGATE_ROOT"),
                        help="if set, every cwd must be inside this directory (PI_DELEGATE_ROOT)")
    parser.add_argument("--extension", action="append", dest="extensions",
                        help="Pi extension file to load for delegated runs, e.g. a path guard; repeatable "
                             f"(PI_DELEGATE_EXTENSIONS, separated by '{os.pathsep}')")
    parser.add_argument("--state-dir", default=env.get("PI_DELEGATE_STATE_DIR"),
                        help="sessions and job records (PI_DELEGATE_STATE_DIR; default: $XDG_STATE_HOME/pi-delegate)")
    args = parser.parse_args(argv)

    extensions = args.extensions
    if extensions is None:
        extensions = [item for item in env.get("PI_DELEGATE_EXTENSIONS", "").split(os.pathsep) if item]
    for extension in extensions:
        if not Path(extension).is_file():
            parser.error(f"extension not found: {extension}")
    root = Path(args.root).expanduser().resolve() if args.root else None
    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    if root and cwd != root and root not in cwd.parents:
        cwd = root
    config = Config(
        pi=args.pi, provider=args.provider,
        consult_model=args.consult_model, delegate_model=args.delegate_model,
        consult_thinking=args.consult_thinking, delegate_thinking=args.delegate_thinking,
        cwd=cwd, root=root, extensions=extensions,
        state_dir=Path(args.state_dir).expanduser() if args.state_dir else default_state_dir(),
    )
    for label, model in (("consult model", config.consult_model), ("delegate model", config.delegate_model)):
        if not valid_model_id(model):
            parser.error(f"invalid {label}: {model}")
    return config


config = Config()
write_lock = threading.Lock()
cancellations: dict[str, threading.Event] = {}
cancellations_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def send_message(payload: dict[str, Any]) -> None:
    with write_lock:
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def text_result(value: Any, *, is_error: bool = False) -> dict[str, Any]:
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def bounded(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[: MAX_OUTPUT_CHARS // 2] + "\n\n[... output truncated ...]\n\n" + text[-MAX_OUTPUT_CHARS // 2:]


def validate_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    if len(value) > MAX_PROMPT_CHARS:
        raise ValueError(f"{label} is longer than {MAX_PROMPT_CHARS} characters")
    return value


def valid_model_id(model: Any) -> bool:
    return isinstance(model, str) and bool(model) and model.replace("-", "").replace(".", "").replace("/", "").replace("_", "").isalnum()


def validate_model(value: Any, default: str) -> str:
    model = default if value in (None, "") else value
    if not valid_model_id(model):
        raise ValueError("model must be a Pi model id such as gpt-6-astra or openai-codex/gpt-6.1-sol")
    return model if "/" in model else f"{config.provider}/{model}"


def validate_thinking(value: Any, default: str) -> str:
    level = default if value in (None, "") else value
    if level not in THINKING_LEVELS:
        raise ValueError(f"thinking must be one of {sorted(THINKING_LEVELS)}")
    return level


def validate_cwd(value: Any) -> Path:
    cwd = config.cwd if value in (None, "") else Path(str(value)).expanduser()
    if not cwd.is_absolute():
        cwd = config.cwd / cwd
    resolved = cwd.resolve()
    if config.root and resolved != config.root and config.root not in resolved.parents:
        raise ValueError(f"cwd must be inside {config.root}")
    if not resolved.is_dir():
        raise ValueError(f"cwd does not exist: {resolved}")
    return resolved


def validate_session(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) > 80 or not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("session must be a short id of letters, digits, '-' or '_'")
    return value


def validate_timeout(value: Any, default: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    if not isinstance(value, int) or not 30 <= value <= maximum:
        raise ValueError(f"timeout_seconds must be an integer from 30 to {maximum}")
    return value


def git_changes(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "status", "--short"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            return "(not a git checkout; changes not listed)"
        lines = result.stdout.strip().splitlines()
        if not lines:
            return "(no uncommitted changes)"
        shown = lines[:80]
        more = f"\n... and {len(lines) - 80} more" if len(lines) > 80 else ""
        return "\n".join(shown) + more
    except (OSError, subprocess.SubprocessError):
        return "(git status unavailable)"


def pi_command(*, model: str, thinking: str, session: str | None, read_only: bool) -> list[str]:
    command = [config.pi, "-p", "--no-skills", "--no-extensions", "--model", model, "--thinking", thinking]
    if read_only:
        command += ["--tools", READ_ONLY_TOOLS]
    else:
        for extension in config.extensions:
            command += ["-e", extension]
    if session:
        config.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        command += ["--session-dir", str(config.session_dir), "--session-id", session]
    else:
        command += ["--no-session"]
    return command


def run_pi(command: list[str], prompt: str, cwd: Path, timeout: int, cancelled: threading.Event,
           on_start: Any = None) -> dict[str, Any]:
    env = {**os.environ, "NODE_NO_WARNINGS": "1"}
    if env.get("HTTPS_PROXY") or env.get("https_proxy"):
        env.setdefault("NODE_USE_ENV_PROXY", "1")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [*command, prompt], cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
    except FileNotFoundError:
        raise ValueError(f"Pi executable not found: {command[0]} (install Pi or set PI_DELEGATE_PI_BIN)") from None
    if on_start:
        on_start(process)
    output: dict[str, str] = {}

    def collect() -> None:
        output["stdout"], output["stderr"] = process.communicate()

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    reason = None
    while reader.is_alive():
        reader.join(0.5)
        if not reader.is_alive():
            break
        if cancelled.is_set():
            reason = "cancelled"
        elif time.monotonic() - started > timeout:
            reason = f"timed out after {timeout}s"
        if reason:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            reader.join(10)
            if reader.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                reader.join(5)
            break
    stdout = (output.get("stdout") or "").strip()
    stderr = "\n".join(
        line for line in (output.get("stderr") or "").splitlines()
        if "Warning" not in line and "trace-warnings" not in line
    ).strip()
    return {
        "ok": reason is None and process.returncode == 0,
        "reason": reason,
        "exitCode": process.returncode,
        "seconds": round(time.monotonic() - started, 1),
        "answer": bounded(stdout),
        "stderr": stderr[-4000:],
    }


def consult(arguments: dict[str, Any], cancelled: threading.Event) -> dict[str, Any]:
    question = validate_text(arguments.get("question"), "question")
    model = validate_model(arguments.get("model"), config.consult_model)
    thinking = validate_thinking(arguments.get("thinking"), config.consult_thinking)
    cwd = validate_cwd(arguments.get("cwd"))
    continuing = validate_session(arguments.get("session"))
    session = continuing or f"consult-{uuid.uuid4().hex[:12]}"
    timeout = validate_timeout(arguments.get("timeout_seconds"), 900, 3600)
    prompt = question if continuing else CONSULT_PREAMBLE + question
    result = run_pi(pi_command(model=model, thinking=thinking, session=session, read_only=True),
                    prompt, cwd, timeout, cancelled)
    result.update({"model": model, "session": session,
                   "note": "Pass this session to consult again to continue the same conversation."})
    return text_result(result, is_error=not result["ok"])


def delegate_parameters(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": validate_text(arguments.get("task"), "task"),
        "model": validate_model(arguments.get("model"), config.delegate_model),
        "thinking": validate_thinking(arguments.get("thinking"), config.delegate_thinking),
        "cwd": validate_cwd(arguments.get("cwd")),
        "session": validate_session(arguments.get("session")),
    }


def delegate(arguments: dict[str, Any], cancelled: threading.Event) -> dict[str, Any]:
    parameters = delegate_parameters(arguments)
    timeout = validate_timeout(arguments.get("timeout_seconds"), 1800, 3600)
    result = run_pi(
        pi_command(model=parameters["model"], thinking=parameters["thinking"],
                   session=parameters["session"], read_only=False),
        DELEGATE_PREAMBLE + parameters["task"], parameters["cwd"], timeout, cancelled,
    )
    result.update({"model": parameters["model"], "session": parameters["session"],
                   "changes": git_changes(parameters["cwd"])})
    return text_result(result, is_error=not result["ok"])


def job_file(job_id: str) -> Path:
    return config.job_dir / f"{job_id}.json"


def save_job(job: dict[str, Any]) -> None:
    config.job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    public = {key: value for key, value in job.items() if not key.startswith("_")}
    temporary = job_file(job["id"]).with_suffix(".tmp")
    temporary.write_text(json.dumps(public, indent=2), encoding="utf-8")
    temporary.replace(job_file(job["id"]))


def delegate_start(arguments: dict[str, Any]) -> dict[str, Any]:
    parameters = delegate_parameters(arguments)
    timeout = validate_timeout(arguments.get("timeout_seconds"), 3600, 4 * 3600)
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    job = {"id": job_id, "state": "running", "model": parameters["model"], "session": parameters["session"],
           "cwd": str(parameters["cwd"]), "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "task": parameters["task"][:500], "_cancel": threading.Event()}
    with jobs_lock:
        jobs[job_id] = job
    save_job(job)

    def work() -> None:
        try:
            result = run_pi(
                pi_command(model=parameters["model"], thinking=parameters["thinking"],
                           session=parameters["session"], read_only=False),
                DELEGATE_PREAMBLE + parameters["task"], parameters["cwd"], timeout, job["_cancel"],
                on_start=lambda process: job.update({"pid": process.pid}),
            )
        except Exception as exc:  # noqa: BLE001 - recorded on the job
            result = {"ok": False, "reason": str(exc)}
        with jobs_lock:
            job.update(result)
            job["state"] = "succeeded" if result["ok"] else ("cancelled" if result["reason"] == "cancelled" else "failed")
            job["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            job["changes"] = git_changes(parameters["cwd"])
            save_job(job)

    threading.Thread(target=work, daemon=True).start()
    return text_result({"job": job_id, "state": "running", "model": parameters["model"],
                        "note": "Check with delegate_status; the job stops if this MCP server exits."})


def load_job(job_id: Any) -> dict[str, Any]:
    if not isinstance(job_id, str) or not job_id.startswith("job-") or not job_id[4:].isalnum():
        raise ValueError("job must be an id returned by delegate_start")
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            return {key: value for key, value in job.items() if not key.startswith("_")}
    try:
        return json.loads(job_file(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unknown job: {job_id}") from exc


def delegate_status(arguments: dict[str, Any]) -> dict[str, Any]:
    job = load_job(arguments.get("job"))
    if job.get("state") == "running" and job["id"] not in jobs:
        job["state"] = "lost"
        job["note"] = "The MCP server that ran this job exited before it finished."
    return text_result(job)


def delegate_cancel(arguments: dict[str, Any]) -> dict[str, Any]:
    job_id = arguments.get("job")
    load_job(job_id)
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("state") != "running":
        return text_result({"job": job_id, "cancelled": False, "reason": "job is not running"})
    job["_cancel"].set()
    return text_result({"job": job_id, "cancelled": True})


def tools() -> list[dict[str, Any]]:
    levels = sorted(THINKING_LEVELS)
    cwd_help = f"Working directory; default {config.cwd}." + (f" Must be inside {config.root}." if config.root else "")
    consult_model = validate_model(None, config.consult_model)
    delegate_model = validate_model(None, config.delegate_model)
    return [
        {
            "name": "consult",
            "description": (
                f"Ask a strong model (default {consult_model}, {config.consult_thinking} thinking) for a second "
                "opinion on a hard problem. It can read this repository (read, grep, find, ls) but not change it. "
                "Include the relevant context and what you want judged. Returns its answer and a session id; pass "
                "the session back to continue the same conversation. Typical calls take one to several minutes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The problem, context, and what you want from the consultant."},
                    "session": {"type": "string", "description": "Session id from an earlier consult to continue it."},
                    "model": {"type": "string", "description": f"Pi model id; default {consult_model}."},
                    "thinking": {"type": "string", "enum": levels, "description": f"Default {config.consult_thinking}."},
                    "cwd": {"type": "string", "description": cwd_help},
                    "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 3600},
                },
                "required": ["question"],
            },
        },
        {
            "name": "delegate",
            "description": (
                f"Hand a well-scoped, routine task to a fast model (default {delegate_model}) that edits this "
                "checkout directly with Pi's tools (read, bash, edit, write). Waits for completion and returns "
                "its summary plus the checkout's uncommitted changes; review them before relying on the work. "
                "Give precise instructions, file paths, and how to verify. For long tasks use delegate_start."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Precise instructions: what to change, where, constraints, and how to verify."},
                    "session": {"type": "string", "description": "Optional session id to keep context across related delegations."},
                    "model": {"type": "string", "description": f"Pi model id; default {delegate_model}."},
                    "thinking": {"type": "string", "enum": levels, "description": f"Default {config.delegate_thinking}."},
                    "cwd": {"type": "string", "description": cwd_help},
                    "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 3600},
                },
                "required": ["task"],
            },
        },
        {
            "name": "delegate_start",
            "description": "Start delegate as a background job and return a job id immediately. Poll with delegate_status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "session": {"type": "string"},
                    "model": {"type": "string"},
                    "thinking": {"type": "string", "enum": levels},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 14400},
                },
                "required": ["task"],
            },
        },
        {
            "name": "delegate_status",
            "description": "Get a background delegation's state, and when finished its answer and the checkout's uncommitted changes.",
            "inputSchema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]},
        },
        {
            "name": "delegate_cancel",
            "description": "Stop a running background delegation.",
            "inputSchema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]},
        },
    ]


def call_tool(name: str, arguments: dict[str, Any], cancelled: threading.Event) -> dict[str, Any]:
    if name == "consult":
        return consult(arguments, cancelled)
    if name == "delegate":
        return delegate(arguments, cancelled)
    if name == "delegate_start":
        return delegate_start(arguments)
    if name == "delegate_status":
        return delegate_status(arguments)
    if name == "delegate_cancel":
        return delegate_cancel(arguments)
    raise ValueError(f"unknown tool: {name}")


def handle_call(request_id: Any, params: dict[str, Any]) -> None:
    key = str(request_id)
    cancelled = threading.Event()
    with cancellations_lock:
        cancellations[key] = cancelled
    try:
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if not isinstance(name, str):
            raise RuntimeError("invalid tools/call parameters")
        send_message({"jsonrpc": "2.0", "id": request_id, "result": call_tool(name, arguments, cancelled)})
    except Exception as exc:  # noqa: BLE001 - every failure becomes a tool error
        send_message({"jsonrpc": "2.0", "id": request_id, "result": text_result(str(exc), is_error=True)})
    finally:
        with cancellations_lock:
            cancellations.pop(key, None)


def handle_request(message: dict[str, Any]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if method == "notifications/cancelled":
        with cancellations_lock:
            event = cancellations.get(str(params.get("requestId")))
        if event:
            event.set()
        return
    if request_id is None:
        return
    if method == "initialize":
        send_message({"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": params.get("protocolVersion") or "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }})
        return
    if method == "ping":
        send_message({"jsonrpc": "2.0", "id": request_id, "result": {}})
        return
    if method == "tools/list":
        send_message({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools()}})
        return
    if method == "tools/call":
        threading.Thread(target=handle_call, args=(request_id, params), daemon=False).start()
        return
    send_message({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}})


def main(argv: list[str] | None = None) -> int:
    global config
    config = parse_config(argv)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if isinstance(message, dict):
                handle_request(message)
        except ValueError:
            continue
        except BrokenPipeError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
