import gradio as gr
import os
import shutil
import uuid
import sqlite3
import datetime
import time
import pandas as pd
import speech_recognition as sr
from groq import Groq
from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import chromadb
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION & MULTILINGUAL SETUP
# ==========================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found! Please check your .env file.")

groq_client = Groq(api_key=api_key)
recognizer = sr.Recognizer()

# Language map for Speech-to-Text (STT) and Text-to-Speech (TTS)
LANG_MAP = {
    "English": {"stt": "en-US", "tts": "en"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"},
    "Marathi": {"stt": "mr-IN", "tts": "mr"},
    "Japanese": {"stt": "ja-JP", "tts": "ja"},
    "Korean": {"stt": "ko-KR", "tts": "ko"}
}

# --- SQLITE DATABASE SETUP ---
DB_NAME = "tutor_database_prod.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS student_logs
                 (timestamp TEXT, student_query TEXT, ai_response TEXT, mode TEXT)''')
    conn.commit()
    conn.close()

def log_to_db(query, response, mode):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO student_logs VALUES (?, ?, ?, ?)", (timestamp, query, response, mode))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

def fetch_logs():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM student_logs ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "student_query", "ai_response", "mode"])

init_db()

# ==========================================
# 2. CORE RAG CLASSES (With Caching)
# ==========================================
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db_temp"):
        self.persist_directory = persist_directory
        self.current_pdf_name = None 
        
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except Exception: pass
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name="demo_docs")
        self.embedding_manager = EmbeddingManager()

    def ingest_pdf(self, pdf_path):
        filename = os.path.basename(pdf_path)
        if self.current_pdf_name == filename:
            return f"⚡ Fast Load: '{filename}' is already active in memory."
            
        try:
            self.collection.delete(where={"source": "upload"})
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks: return "⚠️ No text found in the document."

            

            texts = [doc.page_content for doc in chunks]
            embeddings = self.embedding_manager.generate_embeddings(texts)
            ids = [f"id_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(texts))]
            metadatas = [{"source": "upload"} for _ in range(len(texts))]
            
            self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)
            self.current_pdf_name = filename 
            return f"✅ Success! Indexed {len(chunks)} paragraphs from '{filename}'."
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}"

    def query(self, query_text, n_results=2):
        try:
            query_emb = self.embedding_manager.generate_embeddings([query_text])
            results = self.collection.query(query_embeddings=query_emb.tolist(), n_results=n_results)
            if results['documents'] and results['documents'][0]:
                return " ".join(results['documents'][0])
            return ""
        except: return ""

vector_store = VectorStore()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def text_to_braille(text):
    braille_map = {
        # Lowercase Letters
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
        'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
        'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',
        
        # Numbers (Requires the Braille 'Number Indicator' ⠼ before the letter)
        '1': '⠼⠁', '2': '⠼⠃', '3': '⠼⠉', '4': '⠼⠙', '5': '⠼⠑', 
        '6': '⠼⠋', '7': '⠼⠛', '8': '⠼⠓', '9': '⠼⠊', '0': '⠼⠚',
        
        # Standard Punctuation
        '.': '⠲', ',': '⠂', '?': '⠦', '!': '⠖', ';': '⠆', ':': '⠒',
        '-': '⠤', "'": '⠄', '"': '⠦', '(': '⠣', ')': '⠜', '/': '⠌',
        
        # Basic STEM / Math Symbols
        '+': '⠖', '=': '⠶', '*': '⠔',
        
        # Spacing
        ' ': ' ', '\n': '\n'
    }
    
    result = ""
    # We lower() the text because Grade 1 Braille often ignores capitalization for simplicity
    for char in text.lower(): 
        result += braille_map.get(char, char)
    return result

def save_braille_to_file(braille_text):
    filename = "braille_output.txt"
    with open(filename, "w", encoding="utf-8") as f: f.write(braille_text)
    return os.path.abspath(filename)

def text_to_speech(text, lang_code="en"):
    if not text or len(text.strip()) == 0: return None
    try:
        filename = os.path.abspath(f"response_{uuid.uuid4().hex}.mp3")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(filename)
        time.sleep(0.5) # Prevent Windows file-lock race condition
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            return filename
        return None
    except: return None

def safe_braille_pipeline(text, selected_lang):
    if selected_lang != "English":
        # Translate to English before braille
        translation_prompt = f"Translate this to simple English:\n{text}"
        
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": translation_prompt}],
                temperature=0.2
            )
            text = completion.choices[0].message.content
        except:
            pass

    return text_to_braille(text)

# ==========================================
# 4. CONVERSATIONAL PIPELINE
# ==========================================
def handle_upload(pdf_file):
    if pdf_file: return vector_store.ingest_pdf(pdf_file)
    return "Waiting for study material..."

def process_voice_query(audio_path, chat_history, is_quiz_mode, selected_lang):
    stt_lang = LANG_MAP[selected_lang]["stt"]
    tts_lang = LANG_MAP[selected_lang]["tts"]

    # 1. Voice to Text (Using Selected Language)
    if audio_path:
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                user_query = recognizer.recognize_google(audio_data, language=stt_lang)
        except:
            return None, "Audio unclear.", "Please repeat.", "", None, chat_history
    else:
        return None, "No audio.", "Please speak.", "", None, chat_history

    # 2. RAG Context
    context = vector_store.query(user_query)
    
    # 3. Dynamic Prompting (Quiz vs Standard + Multilingual)
    if is_quiz_mode:
        system_prompt = (
            "You are a Socratic tutor for a visually impaired student. "
            f"Use the Context to give a hint, then ask a guiding question. DO NOT give direct answers. "
            f"CRITICAL: You MUST answer exclusively in {selected_lang}. Do not use English unless defining a specific term."
        )
        mode_label = f"Quiz ({selected_lang})"
    else:
       system_prompt = (
    "You are an expert, empathetic STEM tutor for a visually impaired student. "
    "You must construct your answer using ONLY the provided extracted Context. Do not hallucinate outside information.\n\n"
    "CRITICAL RULES FOR AUDIO ACCESSIBILITY:\n"
    "1. Readability: You are generating text that will be spoken by a text-to-speech engine. Do not use bullet points, asterisks, or bold text.\n"
    "2. Mathematics: You MUST spell out all mathematical equations, symbols, and acronyms exactly as they are spoken. "
    "(e.g., write 'Force equals mass times acceleration' instead of 'F=m*a'. Write 'H 2 O' as 'Water').\n"
    "3. Brevity: Keep the explanation under 3 sentences to prevent audio fatigue.\n"
    f"4. Language: You MUST translate and deliver your final response strictly in {selected_lang}."
     )
    mode_label = f"Tutor ({selected_lang})"
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]: messages.append(msg)
    
    messages.append({"role": "user", "content": f"Context: {context}\nQuestion: {user_query}"})
    
    # 4. LLM Call
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=120
            
        )
        ai_response = completion.choices[0].message.content
    except Exception as e:
        ai_response = f"Error communicating with AI: {str(e)}"

    # 5. Log & Update Memory
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": ai_response})
    log_to_db(user_query, ai_response, mode_label)

    # 6. Threaded Output Generation
    with ThreadPoolExecutor() as executor:
        audio_future = executor.submit(text_to_speech, ai_response, tts_lang)
        braille_future = executor.submit(safe_braille_pipeline, ai_response, selected_lang)
        audio_out = audio_future.result()
        braille_text = braille_future.result()

    braille_file_path = save_braille_to_file(braille_text)
    
    return audio_out, user_query, ai_response, braille_text, braille_file_path, chat_history

# ==========================================
# 5. UNIFIED UI TABS
# ==========================================
theme = gr.themes.Soft(primary_hue="teal", secondary_hue="cyan", font=["Inter", "sans-serif"])

# Embedded JavaScript for Spacebar Push-To-Talk
spacebar_js = """
<script>
let recording = false;

document.addEventListener('keydown', function(e) {
    if (e.code === 'Space' && !recording) {
        e.preventDefault();
        recording = true;

        const audioEls = document.querySelectorAll('gradio-audio');
        audioEls.forEach(el => {
            const btn = el.shadowRoot?.querySelector('button');
            if(btn) btn.click();
        });
    }
});

document.addEventListener('keyup', function(e) {
    if (e.code === 'Space' && recording) {
        e.preventDefault();
        recording = false;

        const audioEls = document.querySelectorAll('gradio-audio');
        audioEls.forEach(el => {
            const btn = el.shadowRoot?.querySelector('button');
            if(btn) btn.click();
        });
    }
});
</script>
"""

with gr.Blocks(title="AI Tutor", theme=theme) as demo:
    chat_memory = gr.State([])

    gr.Markdown("# AI Tutor for Visually Impaired Students")
    gr.Markdown("An accessible, conversational learning platform bridging the gap between students and educational resources.")
    
    # Accessibility HUD
    gr.HTML(f"""
    <div style="text-align: center; padding: 20px; background-color: #0d1117; color: #58a6ff; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="font-size: 24px;" aria-live="polite">🎙️ Voice Tutor Active</h2>
        <p style="font-size: 18px; color: #c9d1d9;">
            <strong>Screen Reader Instructions:</strong><br>
            Press the <strong>TAB</strong> key to navigate to the microphone. <br>
            Alternatively, press the <strong>SPACEBAR</strong> to start and stop recording.
        </p>
    </div>
    {spacebar_js}
    """)
    
    with gr.Row():
        with gr.Row():
            # Left Column: Upload & Voice Input
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Step 1: Provide Material & Settings")
                pdf_in = gr.File(label="Upload Textbook/Notes (PDF)", file_types=[".pdf"])
                upload_status = gr.Textbox(label="Database Status", value="Awaiting document...", interactive=False)
                pdf_in.upload(fn=handle_upload, inputs=[pdf_in], outputs=[upload_status])
                
                with gr.Row():
                    lang_dropdown = gr.Dropdown(choices=list(LANG_MAP.keys()), value="English", label="🌐 Language")
                    quiz_mode_toggle = gr.Checkbox(label="🧠 Enable Quiz Mode", value=False)
                
                gr.Markdown("### 🎤 Step 2: Speak with the Tutor")
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Ask your question here")

            # Right Column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Tutor Output")
                audio_out = gr.Audio(label="Spoken Answer", autoplay=True)
                braille_file_out = gr.File(label="📥 Download Braille (.txt)")
                
                with gr.Accordion("Show Transcripts (For Teachers/Assistants)", open=False):
                    text_out = gr.Textbox(label="Student Asked", interactive=False)
                    ai_out = gr.Textbox(label="Tutor Replied", interactive=False)
                    braille_out = gr.Textbox(label="Braille Simulation", interactive=False)

    with gr.Tab("📊 2. Teacher Dashboard (Analytics)"):
        gr.Markdown("### Student Interaction Logs")
        gr.Markdown("Monitor conversational logs to identify learning gaps and track progress.")
        
        log_table = gr.Dataframe(headers=["Timestamp", "Student Query", "AI Response", "Mode"], interactive=False)
        refresh_btn = gr.Button("🔄 Refresh Data")
        
        demo.load(fn=fetch_logs, outputs=[log_table])
        refresh_btn.click(fn=fetch_logs, outputs=[log_table])

    # Connect Voice Input
    audio_in.stop_recording(
        fn=process_voice_query,
        inputs=[audio_in, chat_memory, quiz_mode_toggle, lang_dropdown],
        outputs=[audio_out, text_out, ai_out, braille_out, braille_file_out, chat_memory]
    )

if __name__ == "__main__":
    demo.launch(share=False)

# import gradio as gr
# import os
# import shutil
# import uuid
# import sqlite3
# import datetime
# import time
# import pandas as pd
# import speech_recognition as sr
# from groq import Groq
# from gtts import gTTS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from concurrent.futures import ThreadPoolExecutor
# import chromadb
# from dotenv import load_dotenv

# # ==========================================
# # 1. CONFIGURATION & DATABASE
# # ==========================================
# load_dotenv()
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# recognizer = sr.Recognizer()

# LANG_MAP = {
#     "English": "en-US",
#     "Hindi (Auto-Detect requires Whisper)": "hi-IN",
#     "Marathi (Auto-Detect requires Whisper)": "mr-IN"
# }

# DB_NAME = "tutor_database_v5.db" # Updated to avoid schema conflicts

# def init_db():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS student_logs
#                  (timestamp TEXT, query TEXT, response TEXT, confidence TEXT, latency REAL)''')
#     conn.commit()
#     conn.close()

# def log_to_db(query, response, confidence, latency):
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     c.execute("INSERT INTO student_logs VALUES (?, ?, ?, ?, ?)", (timestamp, query, response, confidence, latency))
#     conn.commit()
#     conn.close()

# init_db()

# # ==========================================
# # 2. ADVANCED RAG (Confidence & Citations)
# # ==========================================
# class EmbeddingManager:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#     def generate_embeddings(self, texts):
#         return self.model.encode(texts)

# class VectorStore:
#     def __init__(self, persist_directory="./chroma_db_permanent"):
#         self.persist_directory = persist_directory
#         self.client = chromadb.PersistentClient(path=self.persist_directory)
#         self.collection = self.client.get_or_create_collection(name="tutor_docs")
#         self.embedding_manager = EmbeddingManager()

#     def ingest_pdf(self, pdf_path):
#         filename = os.path.basename(pdf_path)
#         existing = self.collection.get(where={"source": filename})
#         if existing and len(existing['ids']) > 0:
#             return f"⚡ Fast Load: '{filename}' cached."
            
#         try:
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
#             chunks = text_splitter.split_documents(documents)
            
#             texts = [doc.page_content for doc in chunks]
#             embeddings = self.embedding_manager.generate_embeddings(texts)
#             ids = [f"id_{uuid.uuid4().hex[:8]}" for _ in texts]
#             metadatas = [{"source": filename} for _ in texts]
            
#             self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)
#             return f"✅ Indexed {len(chunks)} paragraphs from '{filename}'."
#         except Exception as e: return f"❌ Error: {str(e)}"

#     def query(self, query_text, top_k=3):
#         try:
#             query_emb = self.embedding_manager.generate_embeddings([query_text])
#             results = self.collection.query(query_embeddings=query_emb.tolist(), n_results=top_k)
            
#             if results['documents'] and results['documents'][0]:
#                 context = " ".join(results['documents'][0])
#                 source = results['metadatas'][0][0]['source']
#                 distance = results['distances'][0][0]
#                 confidence_score = max(0, min(100, int((1.0 - distance) * 100))) 
#                 return context, source, f"{confidence_score}%"
#             return "", "Unknown", "0%"
#         except: return "", "Unknown", "0%"

# vector_store = VectorStore()

# # ==========================================
# # 3. HELPER FUNCTIONS 
# # ==========================================
# def text_to_braille(text):
#     braille_map = {'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', ' ': ' ', '.': '⠲', ',': '⠂', '\n': '\n'}
#     result = ""
#     for char in text.lower(): result += braille_map.get(char, char)
#     return result

# def save_braille_to_brf(braille_text):
#     filename = "embosser_output.brf"
#     header = "Volume 1\n" + ("-" * 25) + "\n"
#     with open(filename, "w", encoding="utf-8") as f: 
#         f.write(header + braille_text)
#     return os.path.abspath(filename)

# def text_to_speech(text, lang="en"):
#     try:
#         filename = os.path.abspath(f"response_{uuid.uuid4().hex}.mp3")
#         tts = gTTS(text=text, lang=lang, slow=False)
#         tts.save(filename)
#         # 🚨 THE FIX FOR THE AUDIO CRASH: Force the OS to wait 0.5 seconds before Gradio tries to read it
#         time.sleep(0.5) 
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             return filename
#         return None
#     except Exception as e:
#         print(f"TTS Error: {e}")
#         return None

# def fetch_logs():
#     try:
#         conn = sqlite3.connect(DB_NAME)
#         df = pd.read_sql_query("SELECT * FROM student_logs ORDER BY timestamp DESC", conn)
#         conn.close()
#         return df
#     except Exception:
#         return pd.DataFrame(columns=["timestamp", "query", "response", "confidence", "latency"])

# # ==========================================
# # 4. PIPELINE
# # ==========================================
# def handle_upload(pdf_file):
#     if pdf_file: return vector_store.ingest_pdf(pdf_file)
#     return "Waiting..."

# def process_voice_query(audio_path, chat_history, difficulty, lang_choice):
#     start_time = time.time()
#     stt_lang = LANG_MAP[lang_choice]
    
#     # 🚨 THE FIX FOR GRADIO 6.0: Ensure chat_history is a list of dictionaries
#     if not isinstance(chat_history, list):
#         chat_history = []

#     if audio_path:
#         try:
#             with sr.AudioFile(audio_path) as source:
#                 audio_data = recognizer.record(source)
#                 user_query = recognizer.recognize_google(audio_data, language=stt_lang)
#         except: return None, chat_history, "", None, "STT Failed"
#     else: return None, chat_history, "", None, "No Audio"

#     q_lower = user_query.strip().lower()
#     if q_lower in ["stop", "tham", "ruko"]:
#         chat_history.append((user_query, formatted_response))
#         chat_history.append({"role": "assistant", "content": "System: Audio stopped."})
#         return None, chat_history, "", None, "Stopped"

#     # RAG Search
#     context, source, confidence = vector_store.query(user_query)
    
#     if difficulty == "Beginner (5th Grade Level)":
#         level_prompt = "Explain this extremely simply using everyday analogies. Avoid complex jargon."
#     elif difficulty == "Advanced (University Level)":
#         level_prompt = "Provide a highly technical, university-level explanation."
#     else:
#         level_prompt = "Provide a clear, standard explanation."

#     system_prompt = f"You are a tutor. Use the Context to answer. {level_prompt} Keep it under 3 sentences. State your source."
    
#     messages = [{"role": "system", "content": system_prompt}]
    
#     # Load memory properly using Gradio 6.0 Dictionary format
#     for msg in chat_history[-4:]:
#         messages.append({"role": msg["role"], "content": msg["content"]})
        
#     messages.append({"role": "user", "content": f"Context: {context}\nQuestion: {user_query}"})

#     try:
#         completion = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant", messages=messages, temperature=0.3, max_tokens=150
#         )
#         ai_response = completion.choices[0].message.content
#     except Exception as e: ai_response = f"Error: {e}"

#     with ThreadPoolExecutor() as executor:
#         tts_lang = "hi" if "Hindi" in lang_choice else "mr" if "Marathi" in lang_choice else "en"
#         audio_future = executor.submit(text_to_speech, ai_response, tts_lang)
#         braille_future = executor.submit(text_to_braille, ai_response)
#         audio_out = audio_future.result()
#         braille_text = braille_future.result()

#     brf_file = save_braille_to_brf(braille_text)
#     latency = round(time.time() - start_time, 2)
    
#     log_to_db(user_query, ai_response, confidence, latency)
    
#     # Append properly for Gradio 6.0
#     formatted_response = f"[{confidence} Confidence | Source: {source}]\n{ai_response}"
#     chat_history.append((user_query, formatted_response))
#     chat_history.append({"role": "assistant", "content": formatted_response})
    
#     return audio_out, chat_history, braille_text, brf_file, f"Confidence: {confidence}"

# # ==========================================
# # 5. UI WITH DEEP SHADOW DOM JS BYPASS
# # ==========================================
# spacebar_js = """
# function() {
#     document.addEventListener('keydown', function(e) {
#         if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
#             e.preventDefault();
#             const audioEls = document.querySelectorAll('gradio-audio');
#             audioEls.forEach(el => {
#                 const micBtn = el.shadowRoot?.querySelector('button[aria-label="Record audio"]');
#                 if(micBtn) micBtn.click();
#             });
#         }
#     });
# }
# """

# theme = gr.themes.Soft(primary_hue="teal", secondary_hue="cyan")

# with gr.Blocks(title="AI Teaching Aid") as demo:
#     # 🚨 FIX: Force type="messages" to be compatible with Gradio 6.0
#     chat_memory = gr.State([])

#     gr.Markdown("# 🎓 Multimodal AI Teaching Aid")

#     with gr.Tab("👩‍🏫 1. Teacher Dashboard"):
#         with gr.Row():
#             with gr.Column():
#                 pdf_in = gr.File(label="Upload Textbook (PDF)")
#                 upload_status = gr.Textbox(label="Status", interactive=False)
#                 pdf_in.upload(fn=handle_upload, inputs=[pdf_in], outputs=[upload_status])
            
#             with gr.Column():
#                 gr.Markdown("### ⚙️ Student Personalization Profile")
#                 difficulty_dropdown = gr.Dropdown(
#                     choices=["Beginner (5th Grade Level)", "Intermediate (Standard)", "Advanced (University Level)"],
#                     value="Intermediate (Standard)", label="Set Adaptive Difficulty"
#                 )
#                 lang_dropdown = gr.Dropdown(choices=list(LANG_MAP.keys()), value="English", label="Select Language")

#     with gr.Tab("🎙️ 2. Student Interface"):
#         with gr.Row():
#             with gr.Column(scale=1):
#                 gr.Markdown("### 🎤 Speak (Press Spacebar to Toggle Mic)")
#                 audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
#                 conf_out = gr.Textbox(label="Retrieval Confidence", interactive=False)
                
#                 audio_out = gr.Audio(label="Tutor Audio", autoplay=True)
#                 braille_file_out = gr.File(label="📥 Download Embosser File (.brf)")
#                 braille_preview = gr.Textbox(label="Braille Preview", interactive=False)

#             with gr.Column(scale=2):
#                 # 🚨 FIX: Explicitly set type="messages" for Gradio 6.0
#                 chatbot_ui = gr.Chatbot()

#     with gr.Tab("📊 3. Analytics & Logs"):
#         log_table = gr.Dataframe(headers=["Timestamp", "Query", "Response", "Confidence", "Latency (s)"], interactive=False)
#         refresh_btn = gr.Button("🔄 Refresh Data")
#         demo.load(fn=fetch_logs, outputs=[log_table])
#         refresh_btn.click(fn=fetch_logs, outputs=[log_table])

#     audio_in.stop_recording(
#         fn=process_voice_query,
#         inputs=[audio_in, chat_memory, difficulty_dropdown, lang_dropdown],
#         outputs=[audio_out, chatbot_ui, braille_preview, braille_file_out, conf_out]
#     )

# if __name__ == "__main__":
#     demo.launch(share=False, head=spacebar_js, theme=theme)