"""End-to-end tests over stdio with a fake `pi` that echoes its arguments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

FAKE_PI = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, os, sys, time
    prompt = sys.argv[-1]
    if "SLEEP" in prompt:
        time.sleep(30)
    if "WRITE" in prompt:
        open("delegated.txt", "w").write("done")
    if "FAIL" in prompt:
        print("boom", file=sys.stderr)
        sys.exit(3)
    print(json.dumps({"args": sys.argv[1:-1], "prompt": prompt, "cwd": os.getcwd()}))
""")


class Server:
    def __init__(self, *args: str, env: dict[str, str] | None = None, cwd: str | None = None) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "pi_delegate_mcp", *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(SRC), **(env or {})}, cwd=cwd,
        )
        self.next_id = 0

    def send(self, method: str, params: dict | None = None, *, notify: bool = False) -> int | None:
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        request_id = None
        if not notify:
            self.next_id += 1
            request_id = message["id"] = self.next_id
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        return request_id

    def receive(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("server exited: " + self.process.stderr.read())
        return json.loads(line)

    def request(self, method: str, params: dict | None = None) -> dict:
        request_id = self.send(method, params)
        message = self.receive()
        assert message["id"] == request_id, message
        return message

    def call(self, name: str, arguments: dict) -> tuple[bool, object]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})["result"]
        text = result["content"][0]["text"]
        try:
            return result["isError"], json.loads(text)
        except ValueError:
            return result["isError"], text

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait(10)
        self.process.stdout.close()
        self.process.stderr.close()


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.pi = base / "pi"
        self.pi.write_text(FAKE_PI)
        self.pi.chmod(0o755)
        self.root = base / "root"
        self.project = self.root / "project"
        self.project.mkdir(parents=True)
        self.extension = base / "guard.ts"
        self.extension.write_text("export default () => {}\n")
        self.env = {"PI_DELEGATE_PI_BIN": str(self.pi), "PI_DELEGATE_STATE_DIR": str(base / "state")}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def server(self, *args: str, **env: str) -> Server:
        server = Server(*args, env={**self.env, **env}, cwd=str(self.project))
        self.addCleanup(server.close)
        server.request("initialize", {"protocolVersion": "2025-06-18"})
        return server

    def answer(self, payload: dict) -> dict:
        return json.loads(payload["answer"])

    def test_lists_tools_with_configured_defaults(self) -> None:
        server = self.server("--consult-model", "anthropic/claude-x")
        tools = {tool["name"]: tool for tool in server.request("tools/list")["result"]["tools"]}
        self.assertEqual(set(tools), {"consult", "delegate", "delegate_start", "delegate_status", "delegate_cancel"})
        self.assertIn("anthropic/claude-x", tools["consult"]["description"])
        self.assertIn("openai-codex/gpt-6.1-sol", tools["delegate"]["description"])

    def test_consult_is_read_only_and_continues_sessions(self) -> None:
        server = self.server()
        error, first = server.call("consult", {"question": "Is this right?"})
        self.assertFalse(error, first)
        answer = self.answer(first)
        self.assertIn("--tools", answer["args"])
        self.assertEqual(answer["args"][answer["args"].index("--tools") + 1], "read,grep,find,ls")
        self.assertIn("openai-codex/gpt-6-astra", answer["args"])
        self.assertIn("high", answer["args"])
        self.assertTrue(answer["prompt"].startswith("You are being consulted"))
        self.assertEqual(Path(answer["cwd"]).resolve(), self.project.resolve())
        self.assertTrue(first["session"].startswith("consult-"))

        error, second = server.call("consult", {"question": "And now?", "session": first["session"]})
        self.assertFalse(error, second)
        answer = self.answer(second)
        self.assertEqual(answer["prompt"], "And now?")
        self.assertEqual(answer["args"][answer["args"].index("--session-id") + 1], first["session"])

    def test_delegate_loads_extensions_and_reports_changes(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.project)], check=True)
        server = self.server(PI_DELEGATE_EXTENSIONS=str(self.extension))
        error, result = server.call("delegate", {"task": "WRITE a file", "model": "other/model", "thinking": "low"})
        self.assertFalse(error, result)
        answer = self.answer(result)
        self.assertEqual(answer["args"][answer["args"].index("-e") + 1], str(self.extension))
        self.assertIn("other/model", answer["args"])
        self.assertIn("--no-session", answer["args"])
        self.assertTrue(answer["prompt"].startswith("Another coding agent delegated"))
        self.assertIn("delegated.txt", result["changes"])

    def test_root_confines_cwd(self) -> None:
        server = self.server("--root", str(self.root))
        error, message = server.call("consult", {"question": "x", "cwd": "/"})
        self.assertTrue(error)
        self.assertIn("must be inside", message)
        error, result = server.call("consult", {"question": "x", "cwd": str(self.root)})
        self.assertFalse(error, result)

    def test_failures_are_tool_errors(self) -> None:
        server = self.server()
        error, result = server.call("delegate", {"task": "FAIL please"})
        self.assertTrue(error)
        self.assertEqual(result["exitCode"], 3)
        self.assertIn("boom", result["stderr"])
        error, message = server.call("consult", {"question": "x", "thinking": "loud"})
        self.assertTrue(error)
        self.assertIn("thinking must be one of", message)
        error, message = server.call("nope", {})
        self.assertTrue(error)

    def test_missing_pi_is_explained(self) -> None:
        server = self.server(PI_DELEGATE_PI_BIN="/nonexistent/pi")
        error, message = server.call("consult", {"question": "x"})
        self.assertTrue(error)
        self.assertIn("Pi executable not found", message)

    def test_background_job_and_cancel(self) -> None:
        server = self.server()
        error, started = server.call("delegate_start", {"task": "quick"})
        self.assertFalse(error, started)
        for _ in range(50):
            _, status = server.call("delegate_status", {"job": started["job"]})
            if status["state"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(status["state"], "succeeded")

        _, slow = server.call("delegate_start", {"task": "SLEEP"})
        _, cancelled = server.call("delegate_cancel", {"job": slow["job"]})
        self.assertTrue(cancelled["cancelled"])
        for _ in range(50):
            _, status = server.call("delegate_status", {"job": slow["job"]})
            if status["state"] != "running":
                break
            time.sleep(0.2)
        self.assertEqual(status["state"], "cancelled")

    def test_request_cancellation_stops_pi(self) -> None:
        server = self.server()
        request_id = server.send("tools/call", {"name": "consult", "arguments": {"question": "SLEEP"}})
        time.sleep(1)
        server.send("notifications/cancelled", {"requestId": request_id}, notify=True)
        message = server.receive()
        self.assertEqual(message["id"], request_id)
        self.assertEqual(json.loads(message["result"]["content"][0]["text"])["reason"], "cancelled")


if __name__ == "__main__":
    unittest.main()
