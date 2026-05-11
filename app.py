"""
Scientific Paper Summarizer — Streamlit App
Run from Colab:
    !pip install -q streamlit pyngrok textstat
    !streamlit run app.py &
    from pyngrok import ngrok
    print(ngrok.connect(8501))
"""
 
import re
import gc
import torch
import textstat
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)
from peft import PeftModel
from bert_score import score as bert_score_fn
import evaluate

 
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "facebook/bart-large-cnn"
ADAPTER_PATH = "/content/drive/MyDrive/bart-lora-adapter"
KIS_MODEL    = "philippelaban/keep_it_simple"
 
MAX_CHUNK_WORDS    = 400
MIN_SECTION_WORDS  = 60
MATH_DENSITY_LIMIT = 0.12
CHUNK_MIN, CHUNK_MAX = 40, 90
FINAL_MIN, FINAL_MAX = 100, 200
 
DEVICE     = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
 
# ── Text utilities ────────────────────────────────────────────────────────────
 
def clean_text(text: str) -> str:
    text = re.sub(r'@xmath\d+',              '<equation>', text)
    text = re.sub(r'@xcite',                 '<citation>', text)
    text = re.sub(r'\[EQUATION\]',           '<equation>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[EQ\.?\d*\]',          '<equation>', text, flags=re.IGNORECASE)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}',  '',           text)
    text = re.sub(r'\\[a-zA-Z]+',           '',           text)
    text = re.sub(r'\$[^$]{1,80}\$',        '<equation>', text)
    text = re.sub(r'\$\$[^$]+\$\$',         '<equation>', text)
    text = re.sub(r'[A-Z]_\{[^}]+\}',       '',           text)
    text = re.sub(r'[A-Za-z]+_[A-Za-z0-9]+(?=[^a-zA-Z]|$)', '', text)
    text = re.sub(r'[a-zA-Z]+\d*[a-zA-Z]*_[a-zA-Z]+\d*', '', text)
    text = re.sub(r'[A-Z][a-z]+(?:[A-Z][a-z]*){4,}', '', text)
    text = re.sub(r'[ \t]+',   ' ',    text)
    text = re.sub(r'\n{3,}',   '\n\n', text)
    return text.strip()
 
 
def split_sections(text: str) -> list:
    parts = text.split("\n\n")
    return [p.strip() for p in parts if len(p.split()) >= MIN_SECTION_WORDS]
 
 
def chunk_by_sentences(text: str, max_words: int = MAX_CHUNK_WORDS) -> list:
    sentences     = re.split(r'(?<=[.!?])\s+', text)
    chunks        = []
    current       = []
    current_words = 0
    for sent in sentences:
        n = len(sent.split())
        if current_words + n <= max_words:
            current.append(sent)
            current_words += n
        else:
            if current:
                chunks.append(" ".join(current))
            current, current_words = [sent], n
    if current:
        chunks.append(" ".join(current))
    return chunks
 
 
def is_math_heavy(chunk: str) -> bool:
    tokens = chunk.split()
    if not tokens:
        return False
    return (sum(1 for t in tokens if t == '<equation>') / len(tokens)) > MATH_DENSITY_LIMIT
 
 
def build_chunks(text: str) -> list:
    text     = clean_text(text)
    sections = split_sections(text)
    chunks   = []
    for section in sections:
        for chunk in chunk_by_sentences(section):
            if not is_math_heavy(chunk):
                chunks.append(chunk)
    return chunks
 
 
def clean_for_kis(text: str) -> str:
    text = re.sub(r'<equation>|<citation>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
 
# ── Model loading ─────────────────────────────────────────────────────────────
 
@st.cache_resource(show_spinner=False)
def load_bart():
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tok  = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload().to(DEVICE_STR)
    summ  = pipeline("summarization", model=model, tokenizer=tok, device=DEVICE)
    return summ, tok
 
 
@st.cache_resource(show_spinner=False)
def load_kis():
    tok   = AutoTokenizer.from_pretrained(KIS_MODEL)
    model = AutoModelForCausalLM.from_pretrained(KIS_MODEL).to(DEVICE_STR)
    return tok, model
 
 
@st.cache_resource(show_spinner=False)
def load_rouge():
    return evaluate.load("rouge")
 
# ── Summarization ─────────────────────────────────────────────────────────────
 
def run_two_pass_summary(summarizer, article_text: str) -> str:
    chunks = build_chunks(article_text)
    if not chunks:
        return "[No summarizable content found after cleaning.]"
 
    chunks = [c for c in chunks if len(c.split()) >= 50]
    if not chunks:
        return "[All chunks were too short to summarize.]"
 
    chunk_summaries = []
    for chunk in chunks:
        chunk_len = len(chunk.split())
        safe_min  = min(CHUNK_MIN, max(5,  chunk_len // 3))
        safe_max  = min(CHUNK_MAX, max(safe_min + 10, chunk_len))
        try:
            out = summarizer(chunk, min_length=safe_min, max_length=safe_max,
                             do_sample=False, truncation=True)
            chunk_summaries.append(out[0]["summary_text"])
        except Exception:
            continue
 
    if not chunk_summaries:
        return "[No chunks could be summarized.]"
 
    combined     = " ".join(chunk_summaries)
    final_chunks = chunk_by_sentences(combined, max_words=MAX_CHUNK_WORDS)
    final_chunks = [c for c in final_chunks if len(c.split()) >= 50]
    if not final_chunks:
        return combined
 
    final_parts = []
    for chunk in final_chunks:
        chunk_len = len(chunk.split())
        safe_min  = min(FINAL_MIN, max(10, chunk_len // 3))
        safe_max  = min(FINAL_MAX, max(safe_min + 10, chunk_len))
        try:
            out = summarizer(chunk, min_length=safe_min, max_length=safe_max,
                             do_sample=False, truncation=True)
            final_parts.append(out[0]["summary_text"])
        except Exception:
            continue
 
    return " ".join(final_parts) if final_parts else combined
 
 
def simple_summary(bart_summary: str, kis_tok, kis_model) -> str:
    cleaned    = clean_for_kis(bart_summary)
    input_text = cleaned + " " + kis_tok.bos_token
    inputs     = kis_tok(input_text, return_tensors="pt",
                         truncation=True, max_length=512).to(DEVICE_STR)
    with torch.no_grad():
        output_ids = kis_model.generate(
            **inputs, max_new_tokens=150, min_new_tokens=40,
            do_sample=False, no_repeat_ngram_size=3,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return kis_tok.decode(generated, skip_special_tokens=True).strip()
 
# ── Streamlit UI ──────────────────────────────────────────────────────────────
 
st.set_page_config(
    page_title="Scientific Paper Summarizer",
    layout="wide",
)
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    
    st.image("/content/summary.png", use_column_width=True)  # add this as first line inside sidebar
    st.markdown("## About")
    st.markdown(
        "Research papers in AI and machine learning are often too technical "
        "for anyone outside the field. This tool makes them accessible by generating "
        "**two levels of summary** from the same paper"
        " "
        " a **technical summary** that preserves the precise language experts need, "
        " and a **plain-language summary** that explains the core idea to anyone, regardless of background. "
        
    )
    st.divider()
    st.markdown("**Model details**")
    st.caption(f"Base model: `{MODEL_NAME}`")
    st.caption(f"Adapter: `{ADAPTER_PATH}`")
    st.caption(f"Device: `{'GPU' if DEVICE == 0 else 'CPU'}`")
    st.divider()
    mode = st.radio(
        "Output mode",
        ["Technical only", "Plain-language only", "Both side by side"],
        index=2,
    )
 
# Main area 
st.title(" Scientific Paper Summarizer")
st.caption("BART-large + LoRA · ArXiv AI/ML · Two-pass chunking")
 
input_text = st.text_area(
    "Paste paper content",
    height=220,
    placeholder="Paste any AI/ML paper text here — full body, abstract, or a section...",
)
 
run = st.button("Summarize", type="primary")
 
if run:
    if not input_text.strip():
        st.warning("Please paste some paper content first.")
        st.stop()
 
    with st.spinner("Loading models (first run only - cached after)..."):
        bart_summarizer, _ = load_bart()
        kis_tok, kis_model = load_kis()
 
    chunks   = build_chunks(input_text)
    n_chunks = len([c for c in chunks if len(c.split()) >= 50])
 
    progress  = st.progress(0, text="Pass 1 — chunking and summarizing...")
    technical = run_two_pass_summary(bart_summarizer, input_text)
    progress.progress(60, text="Pass 2 — merging chunk summaries...")
 
    plain = None
    if mode in ["Plain-language only", "Both side by side"]:
        plain = simple_summary(technical, kis_tok, kis_model)
 
    progress.progress(100, text="Done!")
    progress.empty()
 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # store in session state so evaluate button works without rerunning
    st.session_state["technical"] = technical
    st.session_state["plain"]     = plain
    st.session_state["n_chunks"]  = n_chunks
    st.session_state["mode"]      = mode
 
# ── Display results ───────────────────────────────────────────────────────────
if "technical" in st.session_state:
    technical = st.session_state["technical"]
    plain     = st.session_state["plain"]
    n_chunks  = st.session_state["n_chunks"]
    mode      = st.session_state["mode"]
 
    st.divider()
 
    if mode == "Both side by side":
        col_t, col_p = st.columns(2)
        with col_t:
            st.markdown("#### Technical summary")
            st.info(technical)
            st.caption(f"{len(technical.split())} words · {n_chunks} chunks processed")
        with col_p:
            st.markdown("#### Plain-language summary")
            st.success(plain)
            st.caption(f"{len(plain.split())} words")
 
    elif mode == "Technical only":
        st.markdown("#### Technical summary")
        st.info(technical)
        st.caption(f"{len(technical.split())} words · {n_chunks} chunks processed")
 
    else:
        st.markdown("#### Plain-language summary")
        st.success(plain)
        st.caption(f"{len(plain.split())} words")
 
    # ── Readability — always shown ─────────────────────────────────────────────
    st.divider()
    st.markdown("#### Readability")
 
    pred_for_metrics = plain if (mode == "Plain-language only" and plain) else technical
 
    fk         = textstat.flesch_kincaid_grade(pred_for_metrics)
    fre        = textstat.flesch_reading_ease(pred_for_metrics)
    fk_simple  = textstat.flesch_kincaid_grade(plain) if plain else None
    fre_simple = textstat.flesch_reading_ease(plain)  if plain else None
 
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("FK Grade (technical)",     f"{fk:.1f}",  help="Grade level needed to read. Lower = easier.")
    r2.metric("Reading Ease (technical)", f"{fre:.1f}", help="0 = hardest, 100 = easiest.")
    if plain:
        delta_fk  = f"{fk_simple  - fk:+.1f} vs technical"
        delta_fre = f"{fre_simple - fre:+.1f} vs technical"
        r3.metric("FK Grade (plain)",     f"{fk_simple:.1f}",  delta=delta_fk,  delta_color="inverse")
        r4.metric("Reading Ease (plain)", f"{fre_simple:.1f}", delta=delta_fre)
 
    # ── Evaluate with reference ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Evaluate against reference abstract")
    st.caption("Paste the paper's original abstract below then click Evaluate to compute ROUGE and BERTScore.")
 
    reference_text = st.text_area(
        "Reference abstract",
        height=120,
        placeholder="Paste the paper's abstract here...",
        key="reference_input"
    )
 
    evaluate_btn = st.button("Evaluate", type="secondary")
 
    if evaluate_btn:
        if not reference_text.strip():
            st.warning("Please paste a reference abstract first.")
        else:
            rouge_metric = load_rouge()
 
            with st.spinner("Computing ROUGE..."):
                rouge = rouge_metric.compute(
                    predictions=[pred_for_metrics],
                    references=[reference_text]
                )
            with st.spinner("Computing BERTScore..."):
                P, R, F1 = bert_score_fn(
                    [pred_for_metrics], [reference_text],
                    lang="en", verbose=False
                )
 
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROUGE-1",      f"{rouge['rouge1']:.3f}", help=">0.35 = good vocabulary overlap")
            m2.metric("ROUGE-2",      f"{rouge['rouge2']:.3f}", help=">0.10 = reasonable phrase match")
            m3.metric("ROUGE-L",      f"{rouge['rougeL']:.3f}", help=">0.30 = decent structural alignment")
            m4.metric("BERTScore F1", f"{F1.mean().item():.3f}", help=">0.85 = strong semantic match")
 
            with st.expander("Metric interpretation guide"):
                st.markdown("""
| Metric | Good threshold | What it measures |
|---|---|---|
| ROUGE-1 | > 0.35 | Vocabulary overlap with reference |
| ROUGE-2 | > 0.10 | Phrase-level match |
| ROUGE-L | > 0.30 | Longest common subsequence |
| BERTScore F1 | > 0.85 | Deep semantic similarity |
| FK Grade | Lower = simpler | US school grade level to read |
| Reading Ease | Higher = simpler | 60–70 standard, 30–50 difficult |
                """)