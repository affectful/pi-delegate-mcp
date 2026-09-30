"""MCP server to consult or delegate to other models through Pi."""

from .server import SERVER_VERSION as __version__, main

__all__ = ["__version__", "main"]
