import os
import re
import json
import time
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import InferenceClient

# Import custom modules
try:
    from pdf_loader import load_pdfs, split_documents
    from embeddings import create_vector_store
    from chatbot import create_chatbot, generate_quiz_from_text
except ImportError:
    st.warning("Custom backend modules missing. Ensure pdf_loader.py, embeddings.py, and chatbot.py exist.")

# -----------------------------------------------------
# INITIAL SETUP & ENV CONFIG
# -----------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

image_client = None
if HF_TOKEN:
    try:
        image_client = InferenceClient(token=HF_TOKEN)
    except Exception:
        pass

USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def generate_visual_prompt(question, answer):
    cleaned_q = re.sub(
        r'(?i)\b(explain|describe|what is|how to|tell me about|list|can you|show|give|detail)\b', 
        '', 
        question
    ).strip()

    clean_answer = re.sub(r'[^\w\s]', ' ', answer[:300])
    words = [
        w for w in clean_answer.split() 
        if len(w) > 4 and w.lower() not in ['this', 'that', 'with', 'from', 'have', 'which']
    ]
    key_terms = ", ".join(words[:6]) if words else cleaned_q

    prompt = (
        f"A clean technical system architecture visual diagram representing {cleaned_q} process workflow. "
        f"Key concepts: {key_terms}. "
        f"Layout style: Professional technical diagram, continuous pipeline, connected node network, "
        f"flat 2D vector graphic, modern UI diagram aesthetics, white background."
    )
    return prompt

def scroll_to_bottom():
    """Injects JavaScript to smoothly scroll the main Streamlit container to the bottom."""
    components.html(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo({
                top: window.parent.document.querySelector('section.main').scrollHeight,
                behavior: 'smooth'
            });
        </script>
        """,
        height=0,
        width=0,
    )

# -----------------------------------------------------
# PAGE CONFIG & DYNAMIC THEME STYLING
# -----------------------------------------------------
st.set_page_config(
    page_title="AI Multi-Document Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------
defaults = {
    "qa_chain": None,
    "processed": False,
    "last_answer": "",
    "last_question": "",
    "last_citation": "",
    "documents": [],
    "chunks": 0,
    "chat_history": [],
    "page": "home",
    "logged_in": False,
    "username": "",
    "generated_image": None,
    "trigger_balloons": False
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def inject_page_theme():
    if st.session_state.page in ["home", "login", "signup"]:
        st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #eaf2ed, #d8e6dc, #c1d7c7) !important;
            }
            .auth-header {
                text-align: center;
                margin-bottom: 25px;
            }
            .auth-title {
                font-size: 28px;
                font-weight: 700;
                color: #1e3a2b;
                margin-top: 10px;
            }
            .auth-sub {
                color: #3b5e4c;
                font-size: 14px;
                margin-top: 5px;
            }
            .stTextInput > div > div > input {
                border-radius: 8px !important;
                border: 1px solid #7a9a85 !important;
                padding: 10px 12px !important;
            }
            .stButton > button[kind="primary"] {
                background-color: #2c4a3e !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                border: none !important;
            }
            .footer {
                text-align: center;
                color: #3b5e4c;
                padding: 20px;
                margin-top: 30px;
                font-size: 13px;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #f3e8ff, #ede9fe, #e6e6fa) !important;
            }
            [data-testid="stSidebar"] {
                background-color: #eae0f8 !important;
                border-right: 1px solid #d8b4fe;
            }
            .source-card {
                background-color: #ffffff;
                border-left: 5px solid #7e22ce;
                padding: 12px 16px;
                border-radius: 8px;
                margin-top: 12px;
                margin-bottom: 12px;
                font-size: 14px;
                color: #581c87;
                box-shadow: 0 4px 10px rgba(126, 34, 206, 0.08);
            }
            .stButton > button[kind="primary"] {
                background-color: #7e22ce !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                border: none !important;
            }
            
            [data-testid="stChatMessage"]:nth-child(odd) {
                background-color: #e6f2ff !important;
                border: 1px solid #b3d7ff !important;
                border-radius: 12px !important;
                padding: 10px 14px !important;
                margin-bottom: 10px !important;
            }
            
            [data-testid="stChatInput"] {
                background: #dbeeff !important;
                border: 2px solid #8ec5ff !important;
                border-radius: 12px !important;
            }

            [data-testid="stChatInput"] textarea {
                background: #dbeeff !important;
                color: #000000 !important;
                border-radius: 12px !important;
            }

            [data-testid="stChatInput"] textarea::placeholder {
                color: #5f7fa3 !important;
            }

            .footer {
                text-align: center;
                color: #6b21a8;
                padding: 20px;
                margin-top: 30px;
                font-size: 13px;
            }
        </style>
        """, unsafe_allow_html=True)

# -----------------------------------------------------
# PAGES
# -----------------------------------------------------
def show_home():
    inject_page_theme()
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <div style="font-size: 70px;">🤖</div>
        <h1 style="font-size: 48px; font-weight: 800; color: #1e3a2b; margin-top: 10px;">
            AI Multi-Document Knowledge Hub
        </h1>
        <p style="font-size: 18px; color: #3b5e4c; max-width: 750px; margin: 15px auto; line-height: 1.6;">
            Transform your static PDF and Word documents into interactive conversations using <b>Google Gemini</b>, <b>LangChain</b>, and <b>FAISS</b> vector search.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1.2, 2])
    with c2:
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state.page = "login"
            st.rerun()

def show_login():
    inject_page_theme()
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="auth-header">
            <div style="font-size: 42px;">🔒</div>
            <div class="auth-title">Welcome Back</div>
            <div class="auth-sub">Enter your details to access your workspace</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="e.g. alex_dev", key="login_user")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In", use_container_width=True, type="primary"):
            users = load_users()
            if not username.strip() or not password.strip():
                st.warning("Please enter both username and password.")
            elif username not in users:
                st.error("Username does not exist.")
            elif users[username]["password"] != password:
                st.error("Incorrect password.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "chatbot"
                st.success("Successfully logged in!")
                time.sleep(0.5)
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅ Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        with c2:
            if st.button("Create Account", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()

def show_signup():
    inject_page_theme()
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="auth-header">
            <div style="font-size: 42px;">📝</div>
            <div class="auth-title">Create an Account</div>
            <div class="auth-sub">Get started with your free document workspace</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Choose a username", key="signup_user")
        email = st.text_input("Email Address", placeholder="name@company.com", key="signup_email")
        password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="signup_pass")
        confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="signup_confirm")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Register Account", use_container_width=True, type="primary"):
            users = load_users()
            if not all([username.strip(), email.strip(), password.strip(), confirm.strip()]):
                st.error("Please fill in all fields.")
            elif username in users:
                st.error("Username already registered.")
            elif "@" not in email or "." not in email:
                st.error("Invalid email address.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                users[username] = {"email": email, "password": password}
                save_users(users)
                st.success("Account created successfully!")
                time.sleep(1)
                st.session_state.page = "login"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Already have an account? Log In", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

# -----------------------------------------------------
# QUIZ SECTION HELPER
# -----------------------------------------------------
def render_quiz_section():
    st.header("🎯 Auto-Generated Document Quiz")
    st.caption("Select an uploaded document to generate an interactive quiz.")

    if not st.session_state.get("documents"):
        st.info("👈 Please upload and process at least one PDF/Word file in the sidebar first.")
        return

    file_names = sorted(list(set(doc.metadata.get("file_name", "Document") for doc in st.session_state.documents)))

    col_select, col_diff, col_num = st.columns([2, 1, 1])
    
    with col_select:
        selected_file = st.selectbox("📄 Select Source Document:", file_names)
    with col_diff:
        difficulty = st.selectbox("⚡ Difficulty:", ["Easy", "Medium", "Hard"])
    with col_num:
        num_questions = st.number_input("❓ Questions:", min_value=3, max_value=10, value=5)

    if st.button("🚀 Generate Interactive Quiz", type="primary"):
        full_text = "\n".join([
            doc.page_content for doc in st.session_state.documents 
            if doc.metadata.get("file_name") == selected_file
        ])

        with st.spinner(f"Analyzing '{selected_file}' and building quiz..."):
            quiz = generate_quiz_from_text(full_text, num_questions, difficulty)
            if quiz:
                st.session_state.current_quiz = quiz
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.success("Quiz generated successfully!")
            else:
                st.error("Failed to parse quiz format. Please try again.")

    if "current_quiz" in st.session_state and st.session_state.current_quiz:
        st.markdown("---")
        st.subheader(f"📝 Quiz: {selected_file}")

        quiz = st.session_state.current_quiz

        with st.form(key="quiz_form"):
            for idx, q in enumerate(quiz):
                st.markdown(f"**Q{idx + 1}: {q['question']}**")
                
                user_choice = st.radio(
                    label=f"Select answer for Q{idx+1}",
                    options=q["options"],
                    key=f"q_{q['id']}",
                    index=None,
                    label_visibility="collapsed"
                )
                st.session_state.user_answers[q["id"]] = user_choice
                st.markdown("<br>", unsafe_allow_html=True)

            submit_quiz = st.form_submit_button("🏆 Submit Answers")

        if submit_quiz:
            st.session_state.quiz_submitted = True
            
            # Check score directly on submission turn to fire balloons ONLY ONCE
            score = sum(1 for q in quiz if st.session_state.user_answers.get(q["id"]) == q["correct_answer"])
            total = len(quiz)
            percentage = (score / total) * 100
            if percentage >= 80:
                st.balloons()

        if st.session_state.get("quiz_submitted"):
            scroll_to_bottom()

            st.markdown("---")
            score = sum(1 for q in quiz if st.session_state.user_answers.get(q["id"]) == q["correct_answer"])
            total = len(quiz)
            percentage = (score / total) * 100

            if percentage >= 80:
                bg_color = "#d1fae5"
                text_color = "#065f46"
                border_color = "#10b981"
                status_msg = "🎉 Outstanding Performance!"
            elif percentage >= 50:
                bg_color = "#dbeafe"
                text_color = "#1e40af"
                border_color = "#3b82f6"
                status_msg = "👍 Good Job!"
            else:
                bg_color = "#fef3c7"
                text_color = "#92400e"
                border_color = "#f59e0b"
                status_msg = "📚 Keep Learning!"

            st.markdown(f"""
                <div style="
                    background-color: {bg_color}; 
                    border: 2px solid {border_color}; 
                    border-radius: 12px; 
                    padding: 20px; 
                    text-align: center; 
                    margin-bottom: 25px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    <h2 style="margin: 0; font-size: 22px; color: {text_color}; font-weight: 700;">
                        {status_msg}
                    </h2>
                    <div style="font-size: 48px; font-weight: 900; color: {text_color}; margin-top: 8px;">
                        {score} <span style="font-size: 32px; font-weight: 600;">/ {total}</span>
                    </div>
                    <div style="font-size: 22px; font-weight: 700; color: {text_color}; margin-top: -5px;">
                        Score Percentage: {percentage:.0f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📋 Detailed Answer Review")
            for idx, q in enumerate(quiz):
                user_ans = st.session_state.user_answers.get(q["id"])
                correct_ans = q["correct_answer"]
                is_correct = user_ans == correct_ans

                with st.expander(f"Q{idx+1}: {q['question']} — {'✅ Correct' if is_correct else '❌ Incorrect'}"):
                    st.write(f"**Your Answer:** {user_ans if user_ans else 'Not Answered'}")
                    st.write(f"**Correct Answer:** {correct_ans}")
                    st.info(f"💡 **Explanation:** {q['explanation']}")

# -----------------------------------------------------
# MAIN CHATBOT APPLICATION
# -----------------------------------------------------
def show_chatbot():
    inject_page_theme()

    # Trigger balloons ONCE when document processing completes
    if st.session_state.get("trigger_balloons"):
        st.balloons()
        st.session_state.trigger_balloons = False

    # Sidebar Setup
    with st.sidebar:
        st.title("📂 Workspace")
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("🔒 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

        st.markdown("---")
        st.subheader("Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or Word files", 
            type=["pdf", "docx", "doc"], 
            accept_multiple_files=True
        )

        process_btn = st.button("🚀 Process Documents", use_container_width=True, type="primary")

        if process_btn and uploaded_files:
            with st.spinner("Processing document chunks & vector indices..."):
                try:
                    documents = load_pdfs(uploaded_files)
                    chunks = split_documents(documents)
                    vector_store = create_vector_store(chunks)
                    qa_chain = create_chatbot(vector_store)

                    st.session_state.qa_chain = qa_chain
                    st.session_state.documents = documents
                    st.session_state.chunks = len(chunks)
                    st.session_state.processed = True
                    st.session_state.trigger_balloons = True
                    st.success("Index ready for queries!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing Error: {e}")

        st.markdown("---")
        st.subheader("📊 Document Statistics")
        file_count = len(set(doc.metadata.get("file_name", "") for doc in st.session_state.documents))
        st.metric("Loaded Files", file_count)
        st.metric("Total Pages / Sections", len(st.session_state.documents))
        st.metric("Indexed Chunks", st.session_state.chunks)

    # Main Dashboard Workspace
    st.title("💬 Chat Workspace")

    if not st.session_state.processed:
        st.info("👈 Upload and process your PDF/Word documents in the sidebar to begin asking questions.")
        return

    # TABS FOR FEATURES
    tab_chat, tab_quiz = st.tabs(["💬 Q&A Chatbot", "🎯 Auto Quiz & Flashcards"])

    with tab_chat:
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if st.session_state.last_citation and st.session_state.last_answer:
                st.markdown(
                    f'<div class="source-card"><b>{st.session_state.last_citation}</b></div>', 
                    unsafe_allow_html=True
                )

            # System Architecture Visual Generator
            if st.session_state.last_answer:
                st.markdown("---")
                st.subheader("📊 Architecture & Workflow Visual")
                col_btn, col_img = st.columns([1, 3])
                
                with col_btn:
                    if st.button(" Generate image ", type="primary"):
                        if not image_client:
                            st.error("HF_TOKEN missing in environment.")
                        else:
                            with st.spinner("Generating diagram..."):
                                try:
                                    enhanced_prompt = generate_visual_prompt(
                                        st.session_state.last_question, 
                                        st.session_state.last_answer
                                    )
                                    generated_img = image_client.text_to_image(
                                        prompt=enhanced_prompt,
                                        model="black-forest-labs/FLUX.1-schnell",
                                        width=768,
                                        height=512
                                    )
                                    st.session_state.generated_image = generated_img
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Generation error: {e}")

                with col_img:
                    if st.session_state.generated_image:
                        st.image(st.session_state.generated_image, caption="Technical Architecture Concept", width=600)

    with tab_quiz:
        render_quiz_section()

    # Chat Input Box at main root to stay strictly pinned at bottom
    if question := st.chat_input("Ask any question from your uploaded documents..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching document database..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": question})
                        answer = response.get("result", "I couldn't find this information in the uploaded PDFs.")
                        docs = response.get("source_documents", [])

                        st.markdown(answer)

                        not_found_phrases = ["no explicit answer found", "not mentioned", "couldn't find", "not found"]
                        is_answer_found = not any(phrase in answer.lower() for phrase in not_found_phrases)

                        if docs and is_answer_found:
                            source_doc = docs[0]
                            file_name = source_doc.metadata.get("file_name", "Document")
                            page_num = source_doc.metadata.get("page", "N/A")
                            page_str = f"Page {page_num}" if page_num != "N/A" else "Main Content"
                            citation_text = f"📌 Citation Source: {file_name} | {page_str}"
                            
                            st.session_state.last_citation = citation_text
                            st.session_state.last_answer = answer
                        else:
                            st.session_state.last_citation = ""
                            st.session_state.last_answer = ""

                        st.session_state.last_question = question
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error fetching response: {e}")

# -----------------------------------------------------
# ROUTER LOGIC
# -----------------------------------------------------
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "login":
    show_login()
elif st.session_state.page == "signup":
    show_signup()
elif st.session_state.page == "chatbot":
    if not st.session_state.logged_in:
        st.session_state.page = "login"
        st.rerun()
    else:
        show_chatbot()

# Footer
st.markdown('<div class="footer">© 2026 AI Multi-Document Assistant • LangChain & Gemini Engine</div>', unsafe_allow_html=True)