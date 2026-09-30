# pi-delegate-mcp

An MCP server that lets Claude Code (or any MCP client) work with other models through [Pi](https://github.com/badlogic/pi-mono):

- **`consult`**: ask a strong model for a second opinion on a hard problem, such as a design choice, a tricky bug, or a review of a plan. It can read the repository (`read`, `grep`, `find`, `ls`) but not change it. Every answer comes with a `session` id; pass it back to continue the conversation. The default is `openai-codex/gpt-6-astra` with high thinking.
- **`delegate`**: hand routine, well-scoped work to a fast model. It edits the checkout directly with Pi's tools, then returns a summary and the uncommitted changes (`git status --short`) so the caller can review them. The default is `openai-codex/gpt-6.1-sol` with medium thinking.
- **`delegate_start`**, **`delegate_status`**, **`delegate_cancel`**: the same as `delegate`, but as a background job.

Pi runs headless (`pi -p`) with skills and extensions disabled. It uses Pi's own login for the provider, for example a ChatGPT/Codex plan through `pi login`. The server uses only the Python standard library and has no dependencies.

## Requirements

- Python 3.10 or newer, and [`uv`](https://docs.astral.sh/uv/) (or `pipx`).
- Pi on `PATH`, logged in to the provider you want to use. Run `pi update --models` if a new model is missing.

## Install

Run it straight from GitHub:

```bash
uvx --from git+https://github.com/affectful/pi-delegate-mcp pi-delegate-mcp --help
```

This repository is private, so `uvx` needs git access to GitHub (an SSH key, or `gh auth setup-git`). You can also install it once:

```bash
uv tool install git+https://github.com/affectful/pi-delegate-mcp    # or: pipx install git+https://...
```

The module is a single self-contained file, so you can also copy `src/pi_delegate_mcp/server.py` anywhere and run it with `python3`.

## Configure clients

### Claude Code

```bash
claude mcp add --scope user pi-delegate -- uvx --from git+https://github.com/affectful/pi-delegate-mcp pi-delegate-mcp
```

Or use `.mcp.json` / `~/.claude.json`:

```json
{
  "mcpServers": {
    "pi-delegate": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/affectful/pi-delegate-mcp", "pi-delegate-mcp"]
    }
  }
}
```

Delegations can take many minutes. Claude Code's default MCP tool timeout may be shorter, so raise it, for example with `export MCP_TOOL_TIMEOUT=3600000`, or use `delegate_start` for long tasks.

### Pi, Antigravity, and other clients

Any client that launches stdio MCP servers works with the same command and args. For Pi's MCP adapter (`~/.pi/agent/mcp.json`), add `"lifecycle": "lazy"` and `"requestTimeoutMs": 3600000` to the entry.

### Teaching the agent the vocabulary

Add something like this to `CLAUDE.md` (or `AGENTS.md`):

```markdown
When I say to **collaborate with Astra**, treat it as an ongoing partnership: share your
understanding and plan with `consult`, weigh Astra's critique against the code, reply in the
same `session` with where you agree or disagree, and iterate until you converge. Then carry out
the agreed approach and tell me what Astra contributed and any point you overruled.
When I say to have **Sol** do something, use `delegate` with precise instructions and review
its changes before reporting.
```

## Options

Each option is a flag or an environment variable. Flags win.

| Flag | Variable | Default |
| --- | --- | --- |
| `--pi` | `PI_DELEGATE_PI_BIN` | `pi` on `PATH` |
| `--provider` | `PI_DELEGATE_PROVIDER` | `openai-codex` (used for model ids without a `provider/` prefix) |
| `--consult-model` | `PI_DELEGATE_CONSULT_MODEL` | `gpt-6-astra` |
| `--delegate-model` | `PI_DELEGATE_DELEGATE_MODEL` | `gpt-6.1-sol` |
| `--consult-thinking` | `PI_DELEGATE_CONSULT_THINKING` | `high` |
| `--delegate-thinking` | `PI_DELEGATE_DELEGATE_THINKING` | `medium` |
| `--cwd` | `PI_DELEGATE_CWD` | the server's working directory (Claude Code starts it in the project) |
| `--root` | `PI_DELEGATE_ROOT` | none; if set, every `cwd` must be inside it |
| `--extension` (repeatable) | `PI_DELEGATE_EXTENSIONS` (`:`-separated) | none; Pi extensions loaded for delegated runs, such as a path guard |
| `--state-dir` | `PI_DELEGATE_STATE_DIR` | `$XDG_STATE_HOME/pi-delegate` (sessions and job records) |

Each tool call can also pass its own `model`, `thinking`, `cwd`, `session`, and `timeout_seconds`.

## Safety notes

- `consult` has only read tools.
- `delegate` runs Pi with its full tool set, including `bash`, in the chosen directory. Use `--root` and a path-guard `--extension` to confine it, and run it only where you would let an agent edit.
- Delegated runs are told not to commit, push, or open pull requests unless the task says to.
- Background jobs belong to the server process. If the client restarts the server, running jobs stop and report `lost`.

## Development

```bash
python3 -m unittest discover -s tests -v
```

The tests use a fake `pi`, so they need no login.
