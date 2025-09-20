import os, re, glob, tempfile, shutil
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
import yt_dlp

# ---------- config ----------
load_dotenv()
OPENAI_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
HAS_OPENAI = bool(OPENAI_KEY)

# lazy-import OpenAI so app works without a key
client = None
if HAS_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
    except Exception:
        HAS_OPENAI = False

st.set_page_config(page_title="YouTube → Transcript → Chunks", layout="wide")

# ---------- helpers ----------
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}")
INDEX_RE     = re.compile(r"^\d+$")

def clean_vtt_or_srt_lines(lines: List[str]) -> Tuple[List[str], str]:
    """
    Remove WebVTT headers, timestamps, <c> tags, and collapse consecutive duplicates.
    Return cleaned lines and a paragraph string.
    """
    cleaned, last_lower = [], None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if TIMESTAMP_RE.match(line):
            continue
        # VTT markup cleanup
        line = re.sub(r"</?c[^>]*>", "", line)     # <c>…</c>
        line = re.sub(r"<[^>]+>", "", line)        # any leftover tags
        if INDEX_RE.match(line):
            continue
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if low != last_lower:
            cleaned.append(line)
            last_lower = low

    paragraph = " ".join(cleaned)
    paragraph = re.sub(r"\s+([,.!?;:])", r"\1", paragraph)
    paragraph = re.sub(r"\s{2,}", " ", paragraph).strip()
    return cleaned, paragraph

def list_available_subs(url: str):
    opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    uploaded = list((info.get("subtitles") or {}).keys())
    auto = list((info.get("automatic_captions") or {}).keys())
    title = info.get("title", "video")
    return title, uploaded, auto


def fetch_transcript_and_chunks():
    """Fetch captions, clean, chunk, and (if available) auto-summarize chunk 1."""
    url = st.session_state.get("yt_url", "").strip()
    if not url:
        return

    with st.spinner("Fetching & preparing transcript…"):   # ⬅️ spinner here
        try:
            title, path = download_captions(
                url,
                lang=st.session_state.get("lang", "en"),
                prefer_uploaded=(st.session_state.get("prefer", "Prefer Auto") == "Prefer Uploaded"),
            )
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            cleaned_lines, paragraph = clean_vtt_or_srt_lines(lines)

            st.session_state["title"] = title
            st.session_state["paragraph"] = paragraph
            st.session_state["raw_lines"] = cleaned_lines

            chunks = split_text_by_words(
                paragraph,
                n_chunks=st.session_state.get("n_chunks", 4),
                max_words_per_chunk=st.session_state.get("max_words", 2500),
            )
            st.session_state["chunks"] = chunks

            if HAS_OPENAI and chunks:
                summary = summarize_with_openai(
                    chunks[0],
                    style=st.session_state.get("style", "Concise (2–3 paragraphs)"),
                    custom=st.session_state.get("custom", ""),
                    model=st.session_state.get("model", "gpt-4o-mini"),
                    max_output_tokens=st.session_state.get("max_out", 700),
                )
                summaries = st.session_state.setdefault("summaries", {})
                summaries[0] = summary
                st.toast("Fetched & summarized chunk 1 ✅")
            else:
                st.toast("Fetched & chunked ✅")

        except Exception as e:
            st.error(str(e))



def download_captions(url: str, lang: str, prefer_uploaded: bool):
    """
    Try preferred source (uploaded or auto); if nothing, try the other.
    Save as VTT (most reliable). Return (title, path_to_file).
    """
    tmp = tempfile.mkdtemp(prefix="ytcap_")
    outtmpl = os.path.join(tmp, "%(title)s.%(id)s.%(ext)s")

    def try_download(uploaded: bool):
        opts = {
            "skip_download": True,
            "writesubtitles": uploaded,
            "writeautomaticsub": (not uploaded),
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
            "convertsubtitles": None,
            "outtmpl": outtmpl,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            try:
                ydl.download([url])
            except yt_dlp.utils.DownloadError:
                pass
        files = glob.glob(os.path.join(tmp, "*.vtt")) + glob.glob(os.path.join(tmp, "*.srt"))
        return info, files

    info, files = try_download(prefer_uploaded)
    if not files:
        info, files = try_download(not prefer_uploaded)

    if not files:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("No captions found. Try a different language code (e.g., en vs en-US) or a different video.")

    title = info.get("title") or "video"
    path = sorted(files)[0]
    return title, path

def split_text_by_words(text: str, n_chunks: int, max_words_per_chunk: int):
    words = text.split()
    if not words:
        return [""] * n_chunks

    if max_words_per_chunk > 0:
        # limit by max words per chunk; derive #chunks from length
        n_chunks = max(1, min(n_chunks, (len(words) + max_words_per_chunk - 1) // max_words_per_chunk))
        size = max_words_per_chunk
    else:
        size = max(1, (len(words) + n_chunks - 1) // n_chunks)

    chunks = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    # ensure exactly n_chunks by merging tail if needed
    if len(chunks) > n_chunks:
        head = chunks[:n_chunks-1]
        tail = " ".join(chunks[n_chunks-1:])
        chunks = head + [tail]
    while len(chunks) < n_chunks:
        chunks.append("")
    return chunks[:n_chunks]

def summarize_with_openai(text: str, style: str, custom: str, model: str, max_output_tokens: int = 700):
    if not (HAS_OPENAI and client):
        return "OpenAI API key not set."
    base_instruction = {
        "Concise (2–3 paragraphs)": "Summarize the text in 2–3 tight paragraphs with the most critical insights. Avoid filler.",
        "Detailed": "Write a comprehensive summary with key points, context, and notable quotes. Keep it factual.",
        "Custom": (custom.strip() or "Summarize clearly and comprehensively.")
    }[style]

    prompt = f"{base_instruction}\n\nText:\n{text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise assistant that summarizes transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=max_output_tokens,
    )
    return resp.choices[0].message.content.strip()

# ---------- UI ----------
st.title("YouTube → Captions → Clean → Chunk → (Optional) Summarize")

with st.sidebar:
    st.header("Controls")
    url = st.text_input("YouTube URL", value="https://www.youtube.com/watch?v=bPI_YEt6RGQ",
        key="yt_url",
        on_change=fetch_transcript_and_chunks )    # ⬅️ Enter triggers fetch + auto-summarize
    # quick helper to see available subs
    if st.button("Show available subtitles"):
        try:
            title, up, auto = list_available_subs(url)
            st.success(f"Title: {title}")
            st.write("Uploaded:", up or "None")
            st.write("Auto:", auto or "None")
        except Exception as e:
            st.error(str(e))

    prefer = st.radio("Caption Source Preference", ["Prefer Auto", "Prefer Uploaded"], index=0)
    lang = st.text_input("Language code (exact)", value="en", help="Use one from the list above, e.g., en, en-US, es, ...")
    n_chunks = st.slider("Number of chunks", 2, 12, 4, 1)
    max_words = st.number_input("Max words per chunk (0 = auto balance)", 0, 10000, 2500, 100)

    st.markdown("---")
    st.subheader("Summaries (Optional)")
    style = st.radio("Style", ["Concise (2–3 paragraphs)", "Detailed", "Custom"], index=0)
    custom = st.text_area("Custom instruction (optional)", height=80, placeholder="e.g., Focus on investment insights and action items.")
    # model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0)
    model = st.selectbox("OpenAI model", ["gpt-4o-mini"], index=0)
    max_out = st.number_input("Max output tokens", 200, 2000, 700, 50)
    if not HAS_OPENAI:
        st.info("No OPENAI_API_KEY found in .env — transcript & chunking still work; summaries will be disabled.")

col1, col2 = st.columns([1,1])


with col1:
    st.subheader("1) Fetch & Clean Transcript")
    if st.button("Fetch"):
        with st.spinner("Fetching captions via yt-dlp & preparing  ……"):
            try:
                fetch_transcript_and_chunks()
                st.toast("Transcript fetched ✅")   # ⬅️ add this line
                # title, path = download_captions(url, lang=lang, prefer_uploaded=(prefer=="Prefer Uploaded"))
                # with open(path, "r", encoding="utf-8") as f:
                #     lines = f.readlines()
                # cleaned_lines, paragraph = clean_vtt_or_srt_lines(lines)
                # st.session_state["title"] = title
                # st.session_state["paragraph"] = paragraph
                # st.session_state["raw_lines"] = cleaned_lines
                # st.success(f"Got captions for: {title}")
            except Exception as e:
                st.error(str(e))

    if "paragraph" in st.session_state:
        st.text_area("Clean Transcript", st.session_state["paragraph"], height=300, key="full_para")
        st.download_button("Download transcript.txt", st.session_state["paragraph"], file_name="transcript.txt")

with col2:
    st.subheader("2) Chunk & (Optional) Summarize")
    if "paragraph" not in st.session_state:
        st.info("Fetch a transcript first.")
    else:
        para = st.session_state["paragraph"]
        chunks = split_text_by_words(para, n_chunks=n_chunks, max_words_per_chunk=max_words)
        st.write(f"**Total words:** {len(para.split()):,} | **Chunks:** {len(chunks)}")
        tabs = st.tabs([f"Chunk {i+1}" for i in range(len(chunks))])
        for i, tab in enumerate(tabs):
            with tab:
                st.caption(f"~{len(chunks[i].split()):,} words")
                st.text_area("Text", chunks[i], height=240, key=f"chunk_ta_{i}")
                st.download_button(f"Download chunk{i+1}.txt", chunks[i], file_name=f"chunk{i+1}.txt")
                if HAS_OPENAI:
                    if st.button(f"Summarize chunk {i+1}", key=f"summarize_{i}"):
                        with st.spinner("Summarizing with OpenAI…"):
                            summary = summarize_with_openai(
                                chunks[i], style=style, custom=custom, model=model, max_output_tokens=max_out
                            )
                        # Save the summary for this chunk so we can render it later (centered)
                        summaries = st.session_state.setdefault("summaries", {})
                        summaries[i] = summary
                        st.success(f"Summary for chunk {i+1} ready below 👇")

# ---------- Centered summaries section (global, full-width) ----------
if st.session_state.get("summaries"):
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 3, 1])   # center
    with c2:
        for idx in sorted(st.session_state["summaries"].keys()):
            st.markdown(f"## 📌 Chunk {idx+1} Summary")
            st.markdown(st.session_state["summaries"][idx])

if st.session_state.get("summaries"):
    if st.button("Clear summaries"):
        st.session_state.pop("summaries", None)
        # st.experimental_rerun()
        # New (correct)
        st.rerun()

