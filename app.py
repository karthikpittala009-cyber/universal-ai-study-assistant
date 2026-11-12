import base64
import io
import os
import re
import time
import requests
import streamlit as st
from fpdf import FPDF
from google import genai
from google.genai import types
from pypdf import PdfReader
from supabase import create_client

st.set_page_config(
    page_title="Cloud AI Study Assistant",
    page_icon="📓",
    layout="wide"
)

# --- HANDWRITTEN NOTEBOOK CSS STYLING ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Patrick+Hand&display=swap');

.handwritten-notebook {
    font-family: 'Patrick Hand', 'Caveat', cursive;
    font-size: 1.35rem;
    line-height: 2.1rem;
    color: #1a237e;
    background-color: #fffef7;
    background-image: 
        linear-gradient(90deg, transparent 49px, #ff8a80 49px, #ff8a80 51px, transparent 51px),
        repeating-linear-gradient(to bottom, transparent 0px, transparent 32px, #cfd8dc 33px);
    padding: 25px 30px 25px 68px;
    border-radius: 10px;
    border: 1px solid #e0dcc8;
    box-shadow: 4px 6px 15px rgba(0,0,0,0.12);
    margin-bottom: 16px;
}

.handwritten-notebook h1,
.handwritten-notebook h2,
.handwritten-notebook h3 {
    font-family: 'Caveat', cursive;
    color: #b71c1c;
    text-decoration: underline;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}

.handwritten-notebook strong {
    color: #0d47a1;
    background-color: #fff59d;
    padding: 0 4px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

st.title("📓 Universal AI Study Assistant (All Subjects • ELI5 + Diagrams)")

# --- AUTO-CONNECT USING STREAMLIT SECRETS ---
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    supa_url = st.secrets["SUPABASE_URL"]
    supa_key = st.secrets["SUPABASE_KEY"]
except Exception:
    st.error("⚠️ Missing keys! Add GEMINI_API_KEY, SUPABASE_URL, and SUPABASE_KEY to .streamlit/secrets.toml.")
    st.stop()

ai_client = genai.Client(api_key=gemini_key)
supabase = create_client(supa_url, supa_key)
st.sidebar.success("☁️ Cloud Storage & AI Connected!")

def get_handwriting_font_path():
    font_path = "PatrickHand-Regular.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 10000:
                with open(font_path, "wb") as f:
                    f.write(resp.content)
        except Exception:
            return None
    return font_path if os.path.exists(font_path) else None

def get_embedding(text: str):
    res = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    return res.embeddings[0].values

def clean_mermaid_code(raw_code: str) -> str:
    code = raw_code.replace("\r\n", "\n").strip()
    lines = []
    for line in code.split("\n"):
        line = re.sub(r"\[([^\]]*)\(([^\)]*)\)([^\]]*)\]", r"[\1- \2\3]", line)
        line = line.replace('"', "").replace(";", "")
        lines.append(line)
    return "\n".join(lines)

def get_mermaid_image_bytes(mermaid_code: str):
    cleaned = clean_mermaid_code(mermaid_code)
    encoded = base64.urlsafe_b64encode(cleaned.encode("utf-8")).decode("ascii")
    img_url = f"https://mermaid.ink/img/{encoded}?bgColor=fffef7&type=png"
    try:
        resp = requests.get(img_url, timeout=15)
        if resp.status_code == 200 and resp.content:
            return resp.content, cleaned
    except Exception:
        pass
    return None, cleaned

def clean_symbols_for_pdf(text: str, has_ttf: bool) -> str:
    # Strip emojis (outside BMP) so FPDF never throws missing-glyph warnings
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    if not has_ttf:
        replacements = {
            "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
            "•": "-", "→": "->", "←": "<-", "↔": "<->", "≥": ">=", "≤": "<="
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "ignore").decode("latin-1")
    return text

class NotebookPDF(FPDF):
    def header(self):
        # Cream notebook paper background
        self.set_fill_color(255, 254, 247)
        self.rect(0, 0, 210, 297, "F")

        # Horizontal ruled notebook lines across the page
        self.set_draw_color(207, 216, 220)
        self.set_line_width(0.25)
        for y in range(16, 290, 8):
            self.line(0, y, 210, y)

        # Red vertical margin line
        self.set_draw_color(255, 138, 128)
        self.set_line_width(0.7)
        self.line(22, 0, 22, 297)

def write_highlighted_handwriting(pdf: FPDF, text: str, font_family: str, line_height: float = 8.0):
    """Renders handwritten text with yellow highlighter behind **bold** keywords."""
    pdf.set_x(pdf.l_margin)
    tokens = re.split(r"(\*\*.*?\*\*)", text)
    for token in tokens:
        if not token:
            continue
        is_bold = token.startswith("**") and token.endswith("**") and len(token) > 4
        clean_tok = token[2:-2] if is_bold else token

        if is_bold:
            pdf.set_text_color(13, 71, 161)
            pdf.set_fill_color(255, 245, 157)
            pdf.set_font(font_family, "U" if font_family == "Handwriting" else "B", 13.5)
            for word in re.findall(r"\S+\s*", clean_tok):
                w = pdf.get_string_width(word)
                if pdf.get_x() + w > pdf.w - pdf.r_margin:
                    pdf.ln(line_height)
                    pdf.set_x(pdf.l_margin)
                pdf.cell(w, line_height, word, fill=True)
        else:
            pdf.set_text_color(26, 35, 126)
            pdf.set_font(font_family, "", 13.5)
            for word in re.findall(r"\S+\s*", clean_tok):
                w = pdf.get_string_width(word)
                if pdf.get_x() + w > pdf.w - pdf.r_margin:
                    pdf.ln(line_height)
                    pdf.set_x(pdf.l_margin)
                pdf.cell(w, line_height, word, fill=False)
    pdf.ln(line_height)

def build_notes_pdf(markdown_text: str, title: str) -> bytes:
    pdf = NotebookPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(26)
    pdf.set_right_margin(15)

    font_path = get_handwriting_font_path()
    if font_path:
        pdf.add_font("Handwriting", "", font_path)
        font_family = "Handwriting"
    else:
        font_family = "Helvetica"

    pdf.add_page()

    # Title in Red Handwritten Ink
    pdf.set_x(pdf.l_margin)
    pdf.set_font(font_family, "U" if font_family == "Handwriting" else "B", 19)
    pdf.set_text_color(183, 28, 28)
    pdf.multi_cell(pdf.epw, 10, clean_symbols_for_pdf(f"Study Notes: {title}", bool(font_path)))
    pdf.ln(4)

    parts = re.split(r"(```mermaid\s*.*?```)", markdown_text, flags=re.DOTALL | re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.lower().startswith("```mermaid"):
            raw_mermaid = re.sub(r"^```mermaid\s*|```$", "", part, flags=re.DOTALL | re.IGNORECASE).strip()
            img_bytes, _ = get_mermaid_image_bytes(raw_mermaid)
            if img_bytes:
                pdf.ln(3)
                img_stream = io.BytesIO(img_bytes)
                pdf.image(img_stream, x=28, w=145)
                pdf.ln(5)
                pdf.set_x(pdf.l_margin)
            else:
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Courier", "", 9)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(pdf.epw, 5, clean_symbols_for_pdf(raw_mermaid, False))
                pdf.ln(2)
        else:
            for line in part.split("\n"):
                clean_line = clean_symbols_for_pdf(line.strip(), bool(font_path))
                if not clean_line:
                    pdf.ln(4)
                    continue

                pdf.set_x(pdf.l_margin)

                if clean_line.startswith("### "):
                    pdf.ln(3)
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font(font_family, "U" if font_family == "Handwriting" else "BU", 16)
                    pdf.set_text_color(183, 28, 28)
                    pdf.multi_cell(pdf.epw, 8.5, clean_line.replace("### ", "").replace("**", ""))
                    pdf.ln(1)
                elif clean_line.startswith("## ") or clean_line.startswith("# "):
                    pdf.ln(4)
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font(font_family, "U" if font_family == "Handwriting" else "BU", 17)
                    pdf.set_text_color(183, 28, 28)
                    pdf.multi_cell(pdf.epw, 9, clean_line.lstrip("# ").replace("**", ""))
                    pdf.ln(1)
                elif clean_line.startswith("- ") or clean_line.startswith("* "):
                    bullet_text = "• " + clean_line[2:]
                    write_highlighted_handwriting(pdf, bullet_text, font_family, line_height=8.0)
                else:
                    write_highlighted_handwriting(pdf, clean_line, font_family, line_height=8.0)

    return bytes(pdf.output())

# --- SIDEBAR: UPLOAD PDF FOR ANY SUBJECT ---
st.sidebar.divider()
st.sidebar.subheader("📤 Upload Any Subject PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload Notes or Question Bank (Any Subject)", type="pdf")

if uploaded_pdf and st.sidebar.button("Save & Index in Cloud"):
    with st.spinner("Uploading PDF to Supabase Storage & indexing text..."):
        raw_bytes = uploaded_pdf.getvalue()
        supabase.storage.from_("study-materials").upload(
            path=uploaded_pdf.name,
            file=raw_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
        reader = PdfReader(uploaded_pdf)
        full_text = "\n".join([p.extract_text() or "" for p in reader.pages])
        chunks = [full_text[i:i+800] for i in range(0, len(full_text), 700)]
        
        rows = []
        for chunk in chunks:
            if chunk.strip():
                rows.append({
                    "file_name": uploaded_pdf.name,
                    "content": chunk,
                    "embedding": get_embedding(chunk)
                })
        if rows:
            supabase.table("document_chunks").insert(rows).execute()
            
        st.sidebar.success(f"☁️ Saved `{uploaded_pdf.name}` ({len(rows)} chunks) to Cloud!")

# Fetch list of uploaded subject files so you can filter by subject
existing_files = []
try:
    file_rows = supabase.table("document_chunks").select("file_name").execute().data or []
    existing_files = sorted(list({r["file_name"] for r in file_rows if r.get("file_name")}))
except Exception:
    pass

# --- MAIN AREA: ASK QUESTIONS ---
st.subheader("✍️ Ask Your Universal Study Tutor")
col1, col2, col3 = st.columns([2.5, 1.3, 1.2])
with col1:
    question = st.text_input("Ask any question or unit from your notes:")
with col2:
    selected_file = st.selectbox("Subject / PDF Filter", ["All Uploaded Subjects"] + existing_files)
with col3:
    detail_level = st.selectbox("Answer Style", ["Fast & Crisp", "Detailed Exam Answers"])

if st.button("Write Handwritten Notes") and question:
    with st.status("Working on your notes...", expanded=True) as status:
        st.write("🔍 Searching your Supabase cloud notes...")
        q_vec = get_embedding(question)
        matches = supabase.rpc("match_chunks", {
            "query_embedding": q_vec,
            "match_count": 8
        }).execute().data or []
        
        if selected_file != "All Uploaded Subjects":
            matches = [m for m in matches if m.get("file_name") == selected_file]

        if not matches:
            status.update(label="No matching notes found!", state="error")
            st.warning("No notes found for that selection. Upload a PDF in the sidebar or choose 'All Uploaded Subjects'.")
            st.stop()

        context = "\n\n".join([f"[Subject File: {m['file_name']}]\n{m['content']}" for m in matches[:5]])
        
        if detail_level == "Fast & Crisp":
            length_rule = "Keep each answer concise with a 2-sentence 5-year-old story and 3 crisp exam bullets."
        else:
            length_rule = "Provide thorough university exam answers with a 5-year-old story, formulas/steps if applicable, and detailed exam bullets."

        prompt = f'''You are a friendly, multi-subject university tutor writing neat handwritten study notes for a student (covering any subject: Engineering, Mathematics, Physics, Computer Science, Electronics, Management, etc.).
Use the retrieved study material below and follow these exact rules for EVERY SINGLE QUESTION:

1. Explain Like I'm 5 (ELI5) First: Start every question with a simple, playful real-world story/analogy a 5-year-old child can picture immediately (using toys, kitchens, postmen, traffic lights, blocks, etc.).
2. Exam-Ready Key Points: Right below the 5-year-old story, write handwritten-style bullet points with **bold keywords** (and clear step-by-step formulas/derivations if it is a math or science question) so the student scores full marks. {length_rule}
3. Mandatory Visual Diagram for Every Question: Include a clean Mermaid diagram (`graph LR` or `graph TD`) wrapped in three backticks + mermaid right after the bullet points for EVERY question—adapt it to the subject (e.g., step-by-step formula/problem-solving flow for Math, circuit/signal flow for Electronics, block/architecture diagram for Computing, or concept map/comparison tree for Theory). Use ONLY simple alphanumeric node IDs and square brackets like `A[Step 1] --> B[Step 2]` with NO parentheses, quotes, or colons inside labels.
4. Format with clear headings (### Question 1: ...).

Retrieved Cloud Material:
{context}

Student Request: {question}'''

        models_to_try = [
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemma-4-31b-it"
        ]
        answer_text = None
        used_model = None
        last_error = ""

        for model_name in models_to_try:
            st.write(f"✍️ Trying `{model_name}`...")
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                answer_text = response.text
                used_model = model_name
                break
            except Exception as e:
                last_error = str(e)
                st.write(f"⚠️ `{model_name}` busy, switching to next free model...")
                time.sleep(1)
                continue

        if answer_text:
            status.update(label=f"✅ Done! Written with {used_model}", state="complete", expanded=False)
            
            parts = re.split(r"(```mermaid\s*.*?```)", answer_text, flags=re.DOTALL | re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.lower().startswith("```mermaid"):
                    raw_mermaid = re.sub(r"^```mermaid\s*|```$", "", part, flags=re.DOTALL | re.IGNORECASE).strip()
                    img_bytes, cleaned_code = get_mermaid_image_bytes(raw_mermaid)
                    if img_bytes:
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.code(cleaned_code, language="mermaid")
                else:
                    st.markdown(f'<div class="handwritten-notebook">\n\n{part}\n\n</div>', unsafe_allow_html=True)

            pdf_bytes = build_notes_pdf(answer_text, question)
            st.download_button(
                label="📥 Download Handwritten Study Notes (PDF + Diagrams)",
                data=pdf_bytes,
                file_name="handwritten_study_notes.pdf",
                mime="application/pdf"
            )
        else:
            status.update(label="Failed to generate", state="error")
            st.error(f"Could not generate answer. Exact API reason: {last_error}")