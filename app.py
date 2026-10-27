import os
import time
import uuid
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()                                   # reads your .env file
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=30000),   # give up after 30 seconds
)
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash"]

SYSTEM_PROMPT = (
    "You are a PDF assistant. Answer ONLY using the context provided. "
    "Do not use outside knowledge. If the answer is not in the context, "
    "reply exactly: 'I could not find this in the PDF.'"
)

def read_pdf(file):
    """Read the text of each page. Returns a list of (page_number, text)."""
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

def make_chunks(pages, size=800, overlap=150):
    """Cut the text into small pieces and remember each piece's page."""
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"text": piece, "page": page_no})
            start += size - overlap
    return chunks

def build_index(chunks):
    """Store the chunks in ChromaDB (it turns them into numbers for searching)."""
    db = chromadb.EphemeralClient()
    col = db.create_collection("pdf_" + uuid.uuid4().hex)
    col.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"]} for c in chunks],
        ids=[f"chunk{i}" for i in range(len(chunks))],
    )
    return col

def answer_question(col, question):
    """Find the best chunks, always add the first pages, then ask Gemini."""
    res = col.query(query_texts=[question], n_results=8)
    docs = res["documents"][0]
    pages = {m["page"] for m in res["metadatas"][0]}

    # the first 2 pages usually have the title, organization and notification number
    first = col.get(where={"page": {"$lte": 2}}, limit=6)
    docs = list(dict.fromkeys(first["documents"] + docs))   # join and remove repeats
    pages.update(m["page"] for m in first["metadatas"])
    pages = sorted(pages)
    context = "\n\n".join(docs)

    last_error = None
    for attempt in range(2):                     # try up to 2 rounds
        for model in MODELS:                     # in each round, try each model
            try:
                print(f"Trying {model} (round {attempt + 1})...", flush=True)
                reply = client.models.generate_content(
                    model=model,
                    contents=f"Context:\n{context}\n\nQuestion: {question}",
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                print(f"Success with {model}", flush=True)
                return reply.text or "I could not get an answer. Please try again.", pages
            except Exception as e:
                last_error = e
                print(f"{model} failed: {str(e)[:120]}", flush=True)
                msg = str(e).lower()
                if any(k in msg for k in ("503", "429", "404", "504", "timed out", "timeout", "deadline")):
                    continue                     # busy, limit, missing or slow: try the next model
                raise                            # any other error: show it
        time.sleep(2)                            # short wait before the next round
    raise last_error

# ---------------- Screen ----------------
st.title("Chat with your PDF")

if not os.getenv("GEMINI_API_KEY"):
    st.error("API key not found. Check your .env file.")
    st.stop()

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("file_name") != uploaded.name:   # index each file only once
        pdf_pages = read_pdf(uploaded)
        chunks = make_chunks(pdf_pages)
        if not chunks:
            st.error("No text found. This may be a scanned PDF.")
            st.stop()
        with st.spinner("Reading your PDF..."):
            st.session_state.col = build_index(chunks)
        st.session_state.first_page_text = pdf_pages[0][1]
        st.session_state.file_name = uploaded.name
        st.session_state.history = []

    with st.expander("Debug: text read from page 1"):
        st.write(st.session_state.first_page_text[:1500] or "NO TEXT FOUND on page 1")

    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    question = st.chat_input("Ask something about the PDF")
    if question:
        st.chat_message("user").write(question)
        try:
            with st.spinner("Thinking..."):
                answer, used_pages = answer_question(st.session_state.col, question)
            full = f"{answer}\n\n*Pages: {used_pages}*"
        except Exception as e:
            full = f"Something went wrong: {e}"
        st.chat_message("assistant").write(full)
        st.session_state.history += [("user", question), ("assistant", full)]