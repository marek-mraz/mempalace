"""Multi-agent swarm concurrency for the pgvector backend.

Regression coverage for the peer-writer latch change: server-mode backends
(pgvector, qdrant) let N MCP servers write the same palace concurrently instead
of latching all but the first into permanent read-only with MCP error -32001.
The flock latch is retained only for local-mode backends (chroma, sqlite_exact)
whose in-process HNSW/FTS cache goes stale on a peer write.

Tests (all runnable without Postgres except the last):

* ``test_pgvector_backend_skips_writer_latch`` — deterministic, in-process:
  the guard skips the flock in server mode, with chroma as the negative control.
* ``test_backend_capability_gating_matches_registry`` — the predicate tracks
  the real ``server_mode`` capability tokens.
* ``test_swarm_cross_process_pgvector_no_peer_writer_error`` — 5 real MCP
  subprocesses sharing one pgvector palace all get to write, zero -32001.
* ``test_swarm_cross_process_chroma_still_latches`` — same swarm on chroma still
  serializes to one writer, peers get -32001 (the guard we must keep).
* ``test_pgvector_swarm_concurrent_kg_add_no_peer_writer_error`` — 5 real MCP
  servers fire the ``kg_add`` mutating tool at once and all writes land in the
  shared KG SQLite. Runs without Postgres (kg_add is SQLite-only).
* ``test_pgvector_swarm_concurrent_add_drawer_no_peer_writer_error`` — the
  full-vector variant firing real ``add_drawer`` calls. Skipped unless
  ``MEMPALACE_PGVECTOR_LIVE_URL`` points at a reachable Postgres+pgvector.
"""

import json
import os
import subprocess
import sys
import threading

import pytest

from mempalace import mcp_server, palace

PEER_WRITER_ERROR_CODE = -32001


def _reset_latch_state(monkeypatch):
    monkeypatch.delenv(mcp_server._MCP_ALLOW_PEER_WRITER_ENV, raising=False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_CM", None)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_FAILED", False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_ERROR", "")


def test_pgvector_backend_skips_writer_latch(monkeypatch):
    """A server-mode backend never consults the flock, so a peer holding it
    does not force this server read-only. A local-mode backend still refuses."""

    def peer_holds_lock(palace_path):
        raise palace.MineAlreadyRunning(f"palace {palace_path} is held by PID 999")

    monkeypatch.setattr(palace, "mine_palace_lock", peer_holds_lock)

    # Server mode (pgvector): the guard short-circuits before the flock, so the
    # mutating tool is allowed and no -32001 refusal is produced.
    _reset_latch_state(monkeypatch)
    monkeypatch.setattr(mcp_server, "_backend_serializes_writers", lambda: True)
    ok, _reason = mcp_server._acquire_mcp_writer_lock()
    assert ok is True
    assert mcp_server._MCP_WRITER_READ_ONLY is False
    assert mcp_server._mcp_peer_writer_refusal(1, "mempalace_add_drawer") is None

    # Local mode (chroma): the flock IS consulted; a live peer latches this
    # server read-only and mutating tools get the -32001 refusal as before.
    _reset_latch_state(monkeypatch)
    monkeypatch.setattr(mcp_server, "_backend_serializes_writers", lambda: False)
    ok, _reason = mcp_server._acquire_mcp_writer_lock()
    assert ok is False
    refusal = mcp_server._mcp_peer_writer_refusal(2, "mempalace_add_drawer")
    assert refusal is not None
    assert refusal["error"]["code"] == PEER_WRITER_ERROR_CODE


def test_backend_capability_gating_matches_registry():
    """The predicate follows the real ``server_mode`` capability tokens."""
    try:
        from mempalace.backends import get_backend_class
    except Exception:  # pragma: no cover - deps not installed
        pytest.skip("backends not importable in this environment")

    expected = {
        "pgvector": True,
        "qdrant": True,
        "chroma": False,
        "sqlite_exact": False,
    }
    for name, serializes in expected.items():
        try:
            caps = get_backend_class(name).capabilities
        except Exception:  # pragma: no cover - optional backend dep missing
            pytest.skip(f"{name} backend not importable")
        assert ("server_mode" in caps) is serializes


# --------------------------------------------------------------------------
# Live end-to-end swarm (opt-in): needs a real Postgres with the pgvector
# extension. Set MEMPALACE_PGVECTOR_LIVE_URL to a DSN, e.g.
#   postgresql://mempalace:mempalace@localhost:5432/mempalace
# --------------------------------------------------------------------------

SWARM_SIZE = 5

# Inline driver run by each subprocess in the cross-process guard tests below.
# It selects the backend from the env, asks the real peer-writer guard whether a
# mutating tool would be refused, prints the verdict, then holds the outcome for
# HOLD seconds so a local-mode flock is still held while its peers try to
# acquire it (real cross-process contention). No Postgres needed: the guard
# decides purely on the OS flock + backend capability, never touching the DB.
_GUARD_DRIVER = """
import os, json, time
# mcp_server redirects fd 1 -> fd 2 at import (os.dup2(2, 1)) and saves the
# real stdout fd as _REAL_STDOUT_FD; write the verdict there so it reaches the
# real stdout the parent reads instead of being mixed into stderr.
from mempalace import mcp_server
ref = mcp_server._mcp_peer_writer_refusal(os.getpid(), "mempalace_add_drawer")
allowed = ref is None
verdict = {"pid": os.getpid(), "allowed": allowed,
           "code": None if allowed else ref["error"]["code"]}
os.write(getattr(mcp_server, "_REAL_STDOUT_FD", 1), (json.dumps(verdict) + "\\n").encode())
time.sleep(float(os.environ.get("HOLD", "0")))
"""


def _spawn_guard_swarm(backend, palace_path, hold=2.0):
    """Launch SWARM_SIZE real subprocesses that hit the guard concurrently.

    Returns the list of parsed verdict dicts. Every subprocess shares one
    palace path, so a local-mode backend's flock is genuinely contended.
    """
    procs = []
    for _ in range(SWARM_SIZE):
        env = dict(os.environ)
        env["MEMPALACE_BACKEND"] = backend
        env["MEMPALACE_BACKEND_EXPLICIT"] = backend
        env["MEMPALACE_PALACE_PATH"] = palace_path
        env["HOLD"] = str(hold)
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", _GUARD_DRIVER],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
        )
    verdicts = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert out.strip(), f"subprocess produced no verdict; stderr:\n{err}"
        verdicts.append(json.loads(out.strip().splitlines()[-1]))
    return verdicts


def test_swarm_cross_process_pgvector_no_peer_writer_error(tmp_path):
    """5 real pgvector MCP processes sharing one palace: all may write, no -32001.

    This is the fix, proven across real OS processes without a live Postgres —
    the guard skips the flock for server-mode backends, so nothing latches
    read-only."""
    verdicts = _spawn_guard_swarm("pgvector", str(tmp_path / "swarm"))

    assert len(verdicts) == SWARM_SIZE
    refused = [v for v in verdicts if not v["allowed"]]
    assert not refused, f"server-mode swarm hit peer-writer refusals: {refused}"
    assert all(v["code"] is None for v in verdicts)


def test_swarm_cross_process_chroma_still_latches(tmp_path):
    """Negative control: a local-mode swarm still serializes to one writer.

    Removing the latch for chroma would reintroduce the in-process HNSW/FTS
    corruption it guards, so peers must still get -32001."""
    verdicts = _spawn_guard_swarm("chroma", str(tmp_path / "swarm"))

    assert len(verdicts) == SWARM_SIZE
    allowed = [v for v in verdicts if v["allowed"]]
    refused = [v for v in verdicts if not v["allowed"]]
    assert allowed, "at least one process should win the writer lock"
    assert refused, "peers should still be latched read-only with -32001"
    assert all(v["code"] == PEER_WRITER_ERROR_CODE for v in refused)


def _mcp_tool_swarm(palace_path, tool, arg_builder, *, extra_env=None):
    """Run SWARM_SIZE real MCP stdio servers and fire one mutating tool on each
    at the same instant. Returns {index: json_rpc_response}.

    Each agent boots its own server, initializes, then blocks on a barrier so
    the tools/call requests land in the same window — the exact race the old
    process-lifetime latch turned into -32001s. ``arg_builder(index)`` returns
    the tool arguments for agent ``index``.
    """
    barrier = threading.Barrier(SWARM_SIZE)
    results = {}

    def worker(index):
        env = dict(os.environ)
        env["MEMPALACE_BACKEND"] = "pgvector"
        env["MEMPALACE_BACKEND_EXPLICIT"] = "pgvector"
        env["MEMPALACE_PALACE_PATH"] = palace_path
        env.update(extra_env or {})

        proc = subprocess.Popen(
            # --palace sets _palace_flag_given so the KG lands in this palace
            # dir instead of the shared default ~/.mempalace KG.
            [sys.executable, "-m", "mempalace.mcp_server", "--palace", palace_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        def send(payload):
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            proc.stdout.readline()  # init response

            barrier.wait(timeout=30)  # release all agents together

            send(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arg_builder(index)},
                }
            )
            line = proc.stdout.readline()
            results[index] = json.loads(line) if line.strip() else None
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(SWARM_SIZE)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    return results


def _assert_no_peer_writer_error(results):
    assert len(results) == SWARM_SIZE, f"only {len(results)} agents responded"
    refused = [
        (i, r)
        for i, r in results.items()
        if r and r.get("error", {}).get("code") == PEER_WRITER_ERROR_CODE
    ]
    assert not refused, f"{len(refused)} agent(s) got -32001 peer-writer refusal: {refused}"
    for i, r in results.items():
        assert r is not None, f"agent {i} got no response"
        assert "error" not in r, f"agent {i} failed: {r['error']}"


def test_pgvector_swarm_concurrent_kg_add_no_peer_writer_error(tmp_path):
    """5 real pgvector MCP servers fire kg_add at once: all succeed, no -32001.

    kg_add writes only the Knowledge Graph SQLite (never the vector store), so
    this exercises the real end-to-end mutating path — removed latch + WAL +
    busy_timeout serializing concurrent cross-process SQLite writes — without a
    live Postgres. Selecting pgvector is what skips the latch."""
    palace_path = str(tmp_path / "kg-swarm")

    results = _mcp_tool_swarm(
        palace_path,
        "mempalace_kg_add",
        lambda i: {"subject": f"agent-{i}", "predicate": "wrote", "object": f"triple-{i}"},
    )
    _assert_no_peer_writer_error(results)

    # Every concurrent kg_add must have actually landed in the shared KG SQLite.
    from mempalace.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=os.path.join(palace_path, "knowledge_graph.sqlite3"))
    try:
        subjects = {row["subject"] for row in kg._conn().execute("SELECT subject FROM triples")}
    finally:
        kg.close()
    assert subjects == {f"agent-{i}" for i in range(SWARM_SIZE)}, (
        f"not all concurrent writes persisted: {subjects}"
    )


def test_pgvector_swarm_concurrent_add_drawer_no_peer_writer_error(tmp_path):
    """Full-vector variant: concurrent add_drawer against a live pgvector.

    add_drawer writes embeddings to Postgres, so this one needs a reachable DB.
    Skipped unless MEMPALACE_PGVECTOR_LIVE_URL is set."""
    dsn = os.environ.get("MEMPALACE_PGVECTOR_LIVE_URL")
    if not dsn:
        pytest.skip("set MEMPALACE_PGVECTOR_LIVE_URL to run the live add_drawer swarm test")

    results = _mcp_tool_swarm(
        str(tmp_path / "drawer-swarm"),
        "mempalace_add_drawer",
        lambda i: {
            "wing": "swarm",
            "room": "concurrency",
            "content": f"verbatim drawer from agent {i}",
            "added_by": f"agent-{i}",
        },
        extra_env={"MEMPALACE_PGVECTOR_DSN": dsn},
    )
    _assert_no_peer_writer_error(results)
