"""PDF page-aware extraction + whole-page retrieval (format_miner).

Covers the pymupdf4llm path only — no palace, no embedding — so the tests
stay fast and hermetic. The MarkItDown fallback and full mine loop are
covered by the existing format_miner tests.
"""

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf4llm")

from mempalace.config import MempalaceConfig  # noqa: E402
from mempalace.format_miner import _extract_pdf_page_chunks, read_pdf_pages  # noqa: E402
from mempalace.mcp_server import TOOLS, tool_get_pdf_pages  # noqa: E402


@pytest.fixture
def pdf(tmp_path):
    """Five-page PDF with distinct, chunk-size-exceeding text per page."""
    topics = ["alpha intro", "beta nutrition", "gamma navigation", "delta defense", "epsilon refs"]
    filler = " Padding sentence so the page clears the minimum chunk size floor."
    path = tmp_path / "handbook.pdf"
    doc = pymupdf.open()
    for t in topics:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 700), t + filler * 2, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def test_page_chunks_carry_pdf_page(pdf):
    chunks, text = _extract_pdf_page_chunks(pdf, MempalaceConfig())
    assert chunks and text
    assert sorted({c["pdf_page"] for c in chunks}) == [1, 2, 3, 4, 5]
    # chunk_index re-numbered globally and unique per file
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    by_page = {c["pdf_page"]: c["content"] for c in chunks}
    assert "gamma navigation" in by_page[3]


def test_read_pdf_pages_range_and_clamp(pdf):
    out = read_pdf_pages(pdf, 2, 4)
    assert out["pages"] == "2-4" and out["total_pages"] == 5
    assert out["filename"] == "handbook.pdf"
    for present in ("beta", "gamma", "delta"):
        assert present in out["text"]
    for absent in ("alpha", "epsilon"):
        assert absent not in out["text"]
    # end omitted = single page; out-of-bounds end clamps to page_count
    assert "gamma" in read_pdf_pages(pdf, 3)["text"]
    assert read_pdf_pages(pdf, 4, 99)["pages"] == "4-5"


def test_read_pdf_pages_errors(pdf, tmp_path):
    with pytest.raises(ValueError):
        read_pdf_pages(tmp_path / "missing.pdf", 1)
    with pytest.raises(ValueError):
        read_pdf_pages(pdf, 9, 12)  # both bounds past the end


def test_broken_pdf_falls_back_to_markitdown_path(tmp_path):
    """Unparseable PDF bytes → (None, None) so extract_text owns the skip."""
    stub = tmp_path / "broken.pdf"
    stub.write_bytes(b"%PDF-1.4 stub")
    assert _extract_pdf_page_chunks(stub, MempalaceConfig()) == (None, None)


def test_mcp_tool_registered_and_error_dict(pdf, tmp_path):
    assert "mempalace_get_pdf_pages" in TOOLS
    assert "error" in tool_get_pdf_pages(str(tmp_path / "missing.pdf"), 1)
    assert "gamma" in tool_get_pdf_pages(str(pdf), 3)["text"]
