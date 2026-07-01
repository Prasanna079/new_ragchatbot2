import os
import re
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi

# ---- load .env so API keys don't need to be typed in the UI ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # if python-dotenv isn't installed, we still fall back to real env vars

# ---- optional deps guarded so app doesn't crash if a package/key is missing ----
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

try:
    from llama_parse import LlamaParse
except ImportError:
    LlamaParse = None

try:
    from groq import Groq
except ImportError:
    Groq = None


# =========================================================
# PAGE CONFIG + STYLING
# =========================================================
st.set_page_config(
    page_title="RAG Playground",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main-header {
    padding: 1.4rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white;
    margin-bottom: 1.4rem;
}
.main-header h1 { margin: 0; font-size: 1.9rem; }
.main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

.chunk-card {
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    background: rgba(127,127,127,0.05);
}
.chunk-meta {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-bottom: 0.35rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-pdf { background:#fee2e2; color:#b91c1c; }
.badge-json { background:#dbeafe; color:#1d4ed8; }
.badge-md { background:#dcfce7; color:#15803d; }
.badge-score { background:#fef3c7; color:#92400e; }

.stButton>button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="main-header">
        <h1>🧩 RAG Playground</h1>
        <p>JSON • Markdown • PDF upload &nbsp;→&nbsp; Chunking (LlamaParse / Direct) &nbsp;→&nbsp; Chunk viewer &nbsp;→&nbsp; BM25 Retrieval + Groq</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================
if "chunks" not in st.session_state:
    st.session_state.chunks = []          # list of dicts: id, source, type, text, created_at
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "bm25_tokens" not in st.session_state:
    st.session_state.bm25_tokens = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = ""

BADGE = {"pdf": "badge-pdf", "json": "badge-json", "markdown": "badge-md"}
ICON = {"pdf": "📄", "json": "📋", "markdown": "📝"}


# =========================================================
# API KEYS — loaded silently from .env, no input fields
# =========================================================
groq_api_key = os.getenv("GROQ_API_KEY", "")
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY", "")

with st.sidebar:
    st.header("🔑 API Keys (.env)")
    if groq_api_key:
        st.success("Groq key loaded ✅")
    else:
        st.error("GROQ_API_KEY not found in .env")

    if llamaparse_api_key:
        st.success("LlamaParse key loaded ✅")
    else:
        st.warning("LLAMA_CLOUD_API_KEY not found (optional, only needed for LlamaParse mode)")

    st.divider()
    st.header("⚙️ Chunking Settings")
    chunk_size = st.slider("Chunk size (words)", 50, 1000, 220, step=10)
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 200, 40, step=10)

    st.divider()
    st.header("🤖 Groq Model")
    groq_model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )

    st.divider()
    if st.button("🗑️ Clear all data", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.bm25 = None
        st.session_state.bm25_tokens = []
        st.session_state.last_results = []
        st.session_state.chat_answer = ""
        st.rerun()

    st.divider()
    total_chunks = len(st.session_state.chunks)
    total_chars = sum(len(c["text"]) for c in st.session_state.chunks)
    st.metric("Total chunks", total_chunks)
    st.metric("Total characters", f"{total_chars:,}")


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def word_chunk(text: str, size: int, overlap: int):
    """Simple sliding-window word based chunker. Word boundary maa katcha, char majh maa haina."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(words):
        piece = words[start : start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
        start += step
    return chunks


def add_chunks(raw_pieces, source_name, source_type):
    now = datetime.now().strftime("%H:%M:%S")
    start_id = len(st.session_state.chunks)
    for i, piece in enumerate(raw_pieces):
        piece = piece.strip()
        if not piece:
            continue
        st.session_state.chunks.append(
            {
                "id": start_id + i,
                "source": source_name,
                "type": source_type,
                "text": piece,
                "created_at": now,
            }
        )


def flatten_json_to_pieces(data):
    """List of records -> ek ek record ek chunk. Dict/aru jasari -> pretty text chunk garni."""
    if isinstance(data, list):
        pieces = []
        for item in data:
            pieces.append(json.dumps(item, ensure_ascii=False, indent=2))
        return pieces
    else:
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        return word_chunk(pretty, chunk_size, chunk_overlap)


def split_markdown_by_headers(md_text: str):
    """## Header haru bata sections banauni, ani each section lai size anusar chunk garni."""
    parts = re.split(r"(?=^#{1,6}\s)", md_text, flags=re.MULTILINE)
    pieces = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        pieces.extend(word_chunk(part, chunk_size, chunk_overlap))
    return pieces if pieces else word_chunk(md_text, chunk_size, chunk_overlap)


def extract_pdf_text_direct(file_path: str) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf install cha na, `pip install pypdf` garnus.")
    reader = pypdf.PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n\n".join(pages_text)


def parse_pdf_with_llamaparse(file_path: str, api_key: str) -> str:
    if LlamaParse is None:
        raise RuntimeError("llama-parse install cha na, `pip install llama-parse` garnus.")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY .env maa set garnus.")
    parser = LlamaParse(api_key=api_key, result_type="markdown")
    documents = parser.load_data(file_path)
    return "\n\n".join(d.text for d in documents)


def build_bm25_index():
    if not st.session_state.chunks:
        st.session_state.bm25 = None
        return
    tokenized = [
        re.findall(r"\w+", c["text"].lower()) for c in st.session_state.chunks
    ]
    st.session_state.bm25_tokens = tokenized
    st.session_state.bm25 = BM25Okapi(tokenized)


def call_groq(query: str, context_chunks, api_key: str, model: str):
    if Groq is None:
        raise RuntimeError("groq package install cha na, `pip install groq` garnus.")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY .env maa set garnus.")
    client = Groq(api_key=api_key)
    context_text = "\n\n---\n\n".join(
        f"[Chunk {c['id']} | source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    system_prompt = (
        "You are a helpful assistant. Answer the user's question ONLY using the "
        "provided context chunks below. If the answer is not in the context, say so clearly. "
        "Keep the answer concise and cite chunk ids you used like [Chunk 2]."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def render_chunk_card(c, score=None):
    badge_cls = BADGE.get(c["type"], "badge-json")
    icon = ICON.get(c["type"], "📦")
    score_html = (
        f'<span class="badge badge-score">score: {score:.3f}</span>' if score is not None else ""
    )
    st.markdown(
        f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                <span class="badge {badge_cls}">{icon} {c['type']}</span>
                {score_html}
                <b>#{c['id']}</b> &nbsp;|&nbsp; source: <i>{c['source']}</i> &nbsp;|&nbsp; {len(c['text'])} chars &nbsp;|&nbsp; {c['created_at']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Show text", expanded=False):
        st.write(c["text"])


# =========================================================
# TABS
# =========================================================
tab_upload, tab_chunks, tab_retrieve = st.tabs(
    ["📤 Upload & Chunk", "🧱 View Chunks", "🔎 Retrieve & Ask"]
)

# ---------------------------------------------------------
# TAB 1: UPLOAD & CHUNK
# ---------------------------------------------------------
with tab_upload:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 JSON Data")
        json_files = st.file_uploader(
            "Upload .json file(s)", type=["json"], key="json_up", accept_multiple_files=True
        )
        json_text_input = st.text_area("...ya JSON yesai paste garnus", height=140, key="json_paste")
        if st.button("➕ Add JSON chunks", key="add_json"):
            added_total = 0
            try:
                if json_files:
                    for jf in json_files:
                        data = json.load(jf)
                        pieces = flatten_json_to_pieces(data)
                        add_chunks(pieces, jf.name, "json")
                        added_total += len(pieces)
                if json_text_input.strip():
                    data = json.loads(json_text_input)
                    pieces = flatten_json_to_pieces(data)
                    add_chunks(pieces, "pasted_json", "json")
                    added_total += len(pieces)
                if added_total == 0:
                    st.warning("JSON file upload garnus ya text paste garnus.")
                else:
                    st.success(f"{added_total} JSON chunks add bhayo ✅")
            except Exception as e:
                st.error(f"JSON parse error: {e}")

        st.divider()
        st.subheader("📝 Markdown Data")
        md_files = st.file_uploader(
            "Upload .md file(s)", type=["md", "markdown", "txt"], key="md_up", accept_multiple_files=True
        )
        md_text_input = st.text_area("...ya Markdown yesai paste garnus", height=140, key="md_paste")
        if st.button("➕ Add Markdown chunks", key="add_md"):
            added_total = 0
            try:
                if md_files:
                    for mf in md_files:
                        text = mf.read().decode("utf-8", errors="ignore")
                        pieces = split_markdown_by_headers(text)
                        add_chunks(pieces, mf.name, "markdown")
                        added_total += len(pieces)
                if md_text_input.strip():
                    pieces = split_markdown_by_headers(md_text_input)
                    add_chunks(pieces, "pasted_markdown", "markdown")
                    added_total += len(pieces)
                if added_total == 0:
                    st.warning("Markdown file upload garnus ya text paste garnus.")
                else:
                    st.success(f"{added_total} Markdown chunks add bhayo ✅")
            except Exception as e:
                st.error(f"Markdown process error: {e}")

    with col2:
        st.subheader("📄 PDF Data")
        pdf_files = st.file_uploader(
            "Upload .pdf file(s)", type=["pdf"], key="pdf_up", accept_multiple_files=True
        )
        parse_mode = st.radio(
            "PDF process kasari garne?",
            ["Direct chunk (pypdf, no API)", "Parse with LlamaParse (better structure)"],
            index=0,
        )
        if st.button("➕ Add PDF chunks", key="add_pdf"):
            if not pdf_files:
                st.warning("PDF file(s) upload garnus.")
            else:
                added_total = 0
                for pdf_file in pdf_files:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp.write(pdf_file.read())
                            tmp_path = tmp.name

                        if parse_mode.startswith("Parse"):
                            with st.spinner(f"LlamaParse le {pdf_file.name} parse gardai cha..."):
                                full_text = parse_pdf_with_llamaparse(tmp_path, llamaparse_api_key)
                            pieces = split_markdown_by_headers(full_text)
                        else:
                            with st.spinner(f"{pdf_file.name} bata direct text nikaldai cha..."):
                                full_text = extract_pdf_text_direct(tmp_path)
                            pieces = word_chunk(full_text, chunk_size, chunk_overlap)

                        os.unlink(tmp_path)

                        if not pieces:
                            st.warning(f"{pdf_file.name} bata text kei niklena (scanned image PDF ki?).")
                        else:
                            add_chunks(pieces, pdf_file.name, "pdf")
                            added_total += len(pieces)
                    except Exception as e:
                        st.error(f"{pdf_file.name} process error: {e}")

                if added_total:
                    st.success(f"{added_total} PDF chunks add bhayo ✅ ({len(pdf_files)} file(s) baata)")

    st.divider()
    st.caption(
        "Tip: sabai upload garisake pachi **'View Chunks'** tab ma herna sakincha, "
        "ani index build garera **'Retrieve & Ask'** tab bata query garna sakincha. "
        "Multiple files ek pallai select garna sakincha (Ctrl/Cmd + click)."
    )

# ---------------------------------------------------------
# TAB 2: VIEW CHUNKS
# ---------------------------------------------------------
with tab_chunks:
    if not st.session_state.chunks:
        st.info("Halsamma kunai chunk add bhayeko chaina. 'Upload & Chunk' tab bata data halnus.")
    else:
        df = pd.DataFrame(st.session_state.chunks)

        colf1, colf2, colf3 = st.columns([1, 1, 2])
        with colf1:
            type_filter = st.multiselect(
                "Type filter", options=sorted(df["type"].unique()), default=list(df["type"].unique())
            )
        with colf2:
            source_filter = st.multiselect(
                "Source filter", options=sorted(df["source"].unique()), default=list(df["source"].unique())
            )
        with colf3:
            search_kw = st.text_input("🔍 Text ma search (keyword)")

        filtered = df[df["type"].isin(type_filter) & df["source"].isin(source_filter)]
        if search_kw.strip():
            filtered = filtered[filtered["text"].str.contains(search_kw, case=False, na=False)]

        st.caption(f"{len(filtered)} / {len(df)} chunks dekhaudai")

        for _, row in filtered.iterrows():
            render_chunk_card(row.to_dict())

        with st.expander("📊 Raw table view"):
            st.dataframe(
                filtered[["id", "type", "source", "created_at"]].assign(
                    length=filtered["text"].str.len()
                ),
                use_container_width=True,
            )

# ---------------------------------------------------------
# TAB 3: RETRIEVE & ASK
# ---------------------------------------------------------
with tab_retrieve:
    if not st.session_state.chunks:
        st.info("Pahila chunk add garnus, ani yeta index build garera query garnus.")
    else:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("🔨 Build / Refresh Index", use_container_width=True):
                build_bm25_index()
                st.success("BM25 index ready ✅")

        if st.session_state.bm25 is None:
            st.warning("Query garnu agadi 'Build / Refresh Index' click garnus.")
        else:
            with st.form("query_form"):
                query = st.text_input("Your question")
                top_k = st.slider("Top-K retrieved chunks", 1, 10, 4)
                use_groq = st.checkbox("Groq bata final answer pani generate garne", value=True)
                submitted = st.form_submit_button("🔎 Search")

            if submitted and query.strip():
                tokenized_query = re.findall(r"\w+", query.lower())
                scores = st.session_state.bm25.get_scores(tokenized_query)
                ranked = sorted(
                    zip(st.session_state.chunks, scores), key=lambda x: x[1], reverse=True
                )[:top_k]
                st.session_state.last_results = ranked
                st.session_state.chat_answer = ""

                if use_groq:
                    try:
                        with st.spinner("Groq bata answer banaudai cha..."):
                            answer = call_groq(
                                query,
                                [c for c, s in ranked],
                                groq_api_key,
                                groq_model,
                            )
                        st.session_state.chat_answer = answer
                    except Exception as e:
                        st.error(f"Groq error: {e}")

            if st.session_state.chat_answer:
                st.subheader("🤖 Answer")
                st.markdown(st.session_state.chat_answer)
                st.divider()

            if st.session_state.last_results:
                st.subheader("📥 Retrieved chunks")
                for c, s in st.session_state.last_results:
                    render_chunk_card(c, score=float(s))