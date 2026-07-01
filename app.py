
import streamlit as st
from datetime import datetime
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def flatten_html(s):
    return " ".join(line.strip() for line in s.splitlines() if line.strip())

@st.cache_resource
def download_nltk_packages():
    for res in ['punkt','punkt_tab','stopwords','wordnet']:
        try: nltk.download(res, quiet=True)
        except: pass

download_nltk_packages()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t not in string.punctuation)

FAQ_CATEGORIES = {
    "Getting Started": {
        "How do I sign up for the premium membership tier?": "Access the subscription tier management route within your primary dashboard space and trigger the upgrade billing transaction flow.",
        "How do I reset my account password?": "Navigate to the login portal and click on 'Forgot Password' to receive a secure password reset link.",
        "How can I delete my data profile permanently?": "Navigate to Account Settings Management, locate the Data Privacy node, and execute the 'Purge Profile Request' command sequence.",
        "Can I merge two separate user profiles?": "Data structures restrict profile merging directly due to structural integrity protocols. Contact identity governance teams for manual payload migrations.",
        "Where can I find my unique activation token key?": "Your unique activation keys are securely dispatched inside your initial welcome email logs or stored inside the profile metadata matrix."
    },
    "Features": {
        "Do you offer corporate or bulk purchase discounts?": "Yes, enterprise discounts are available for volume configurations. Please route your request to our commercial division via sales@platform.com.",
        "Do you issue official commercial invoice documents?": "Yes, PDF invoice sheets generate immediately post-checkout approval execution. Access historic copies inside your order logging matrices anytime.",
        "Why is my promotional discount code invalid?": "Coupons fail structural validation loops if execution timestamps expire, cart value index checks drop low, or item category arrays mismatch.",
        "Is there an offline operation sync functionality mode?": "Yes, local storage client caches capture inputs during drops. Systems trigger background synchronization payloads once network pipelines restore active state.",
        "Can I export analytical system reporting matrices?": "Analytical data tracking engines allow direct extraction into CSV or structured JSON format nodes via the primary reporting interface."
    },
    "Technical Support": {
        "How can I track my order shipment?": "Once your order ships, an email containing a unique tracking number and link will be dispatched to you.",
        "What should I do if I receive a damaged product?": "Please transmit photographic evidence of the damaged unit to support@platform.com within 48 hours of delivery to initiate an instant replacement protocol.",
        "Can I change my delivery shipping address post-purchase?": "Shipping modifications are restricted once checkout logs hit processing state. Try targeting support flags within 30 minutes of order placement.",
        "How can I contact corporate client support?": "Our customer support division is available 24/7 via the live chat interface or email support@platform.com.",
        "What browsers match your interface framework requirements?": "Web portals are highly optimized for Chromium-based architectures, modern Safari builds, and Firefox engine versions."
    },
    "Payment Plans": {
        "What payment methods do you accept?": "We accept all major credit/debit cards, PayPal, Apple Pay, and secure bank wire transfers.",
        "How long does the refund process take?": "Once we receive your returned item, the inspect protocol takes 2-3 business days. The refund will reflect in your account within 5-7 working days.",
        "Do you support Cash on Delivery (COD)?": "Cash on Delivery structures are supported exclusively across select regional geographic clusters. Check availability token metrics at individual checkouts.",
        "Are tax assessments included in the initial product price?": "Product prices exclude tax variables. Applicable regional tax matrices are calculated dynamically onto checkout totals before authorization requests.",
        "What happens if an authorized payment fails mid-session?": "Session tokens trigger an automatic rollover rollback sequence. Any pending balance locks clear natively via your card networks within 7 business phases."
    }
}

FLAT_FAQ_DATASET = {}
for cat in FAQ_CATEGORIES.values(): FLAT_FAQ_DATASET.update(cat)
faq_questions = list(FLAT_FAQ_DATASET.keys())
faq_answers   = list(FLAT_FAQ_DATASET.values())
preprocessed_faq_questions = [preprocess_text(q) for q in faq_questions]

def get_bot_response(user_query):
    pq = preprocess_text(user_query)
    if not pq.strip():
        return "I am sorry, I didn't catch that. Could you please rephrase?", 0.0
    corpus = preprocessed_faq_questions + [pq]
    vec = TfidfVectorizer()
    mat = vec.fit_transform(corpus)
    scores = cosine_similarity(mat[-1], mat[:-1])[0]
    idx = scores.argmax()
    score = scores[idx]
    if score >= 0.20:
        return faq_answers[idx], round(float(score)*100, 1)
    return "I am unable to find an exact match. Please contact support@platform.com.", round(float(score)*100, 1)

st.set_page_config(page_title="CodeAlpha FAQ Assistant", page_icon="🤖", layout="wide")

defaults = {
    "speech_greeted": False,
    "messages": [{"role":"assistant","content":"How can we help you? Choose from the quick FAQ cards below or type your own question. ⚡","time":datetime.now().strftime("%I:%M %p"),"confidence":None}],
    "bot_talking": False,
    "latest_reply": "How can we help you?",
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #020c18 0%, #041428 40%, #051a2e 70%, #020c18 100%) !important;
    perspective: 1800px;
    overflow-x: hidden;
}
.stApp::before {
    content: "";
    position: fixed; top:-60%; left:-60%; width:220vw; height:220vh;
    background-image:
        linear-gradient(rgba(6,182,212,0.11) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6,182,212,0.11) 1px, transparent 1px),
        linear-gradient(rgba(20,184,166,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20,184,166,0.05) 1px, transparent 1px);
    background-size: 60px 60px, 60px 60px, 20px 20px, 20px 20px;
    transform: rotateX(58deg) rotateZ(-42deg) translateZ(-60px);
    z-index: 0; pointer-events: none;
    animation: gridDrift 45s linear infinite;
}
@keyframes gridDrift { to { background-position: 600px 600px, 600px 600px, 200px 200px, 200px 200px; } }

.matrix-wrapper {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    margin-top: 8px; position: relative; z-index: 1;
}
.chat-terminal-3d {
    background: linear-gradient(160deg, rgba(13,18,58,0.85) 0%, rgba(20,26,80,0.9) 60%, rgba(13,18,58,0.85) 100%);
    backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
    border: 1.5px solid rgba(139,92,246,0.5);
    border-radius: 28px;
    padding: 28px 32px;
    width: 100%; max-width: 660px; height: 430px;
    display: flex; flex-direction: column; position: relative;
    box-shadow: 0 30px 60px rgba(0,0,0,0.6), 0 0 80px rgba(139,92,246,0.14);
}

.chat-scroller { flex:1; overflow-y:auto; padding-right:5px; display:flex; flex-direction:column; gap:10px; }
.chat-scroller::-webkit-scrollbar { width:3px; }
.chat-scroller::-webkit-scrollbar-thumb { background:rgba(139,92,246,0.4); border-radius:10px; }

.bot-stage {
    position:absolute; top:-116px; right:28px; width:124px; height:124px;
}
.bot-float { width:100%; height:100%; animation:botFloat 3s ease-in-out infinite; filter:drop-shadow(0 0 14px rgba(139,92,246,0.55)); }
@keyframes botFloat { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-8px);} }

.bot-mouth.talking { 
    animation: mouthTalk 0.25s ease-in-out infinite alternate !important; 
    transform-origin: 75px 65px !important; 
}
@keyframes mouthTalk { 
    0% { transform: scaleY(0.4); fill: #ff4757; } 
    100% { transform: scaleY(2.2); fill: #00dfa2; } 
}

.bubble-row { display:flex; flex-direction:column; max-width:84%; }
.bubble-row.user   { align-self:flex-end;  align-items:flex-end; }
.bubble-row.assistant { align-self:flex-start; align-items:flex-start; }
.bubble-node { padding:11px 16px; border-radius:18px; font-size:13.5px; line-height:1.5; }
.bubble-node.user      { background:linear-gradient(135deg,#6d28d9,#4f46e5); color:#fff; border-bottom-right-radius:4px; }
.bubble-node.assistant { background:rgba(255,255,255,0.05); color:#e2e8f0; border:1px solid rgba(139,92,246,0.22); border-bottom-left-radius:4px; }
.bubble-meta { font-size:9.5px; color:#5a6e85; margin-top:3px; padding:0 4px; }

form[data-testid="stForm"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 16px !important; padding: 4px 10px !important; margin-top:10px; width: 100%; max-width: 660px;
}
form[data-testid="stForm"] input { background:transparent !important; color:#e2e8f0 !important; border:none !important; font-size:13.5px !important; }
form[data-testid="stForm"] button { background:linear-gradient(135deg,#7c3aed,#4f46e5) !important; color:#fff !important; border-radius:10px !important; font-weight:700 !important; }

div.stButton > button {
    background: linear-gradient(145deg, rgba(4,28,48,0.92), rgba(5,38,58,0.95)) !important;
    color: #67e8f9 !important;
    border: 1px solid rgba(6,182,212,0.30) !important;
    border-radius: 16px !important;
    padding: 10px 13px !important; font-size: 12px !important; min-height: 60px !important;
}
div.stButton > button:hover {
    background: linear-gradient(145deg, rgba(6,182,212,0.22), rgba(20,184,166,0.18)) !important;
    color: #fff !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Native JavaScript Audio Engine Bridge Hook
st.markdown(
    """
    <script>
    window.nativeSpeechTrigger = function(rawText) {
        if('speechSynthesis' in window){
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance(rawText);
            utterance.rate = 1.0;
            utterance.pitch = 1.1;
            
            utterance.onstart = function() {
                var mouth = document.querySelector('.bot-mouth');
                if(mouth) mouth.classList.add('talking');
            };
            utterance.onend = function() {
                var mouth = document.querySelector('.bot-mouth');
                if(mouth) mouth.classList.remove('talking');
            };
            utterance.onerror = function() {
                var mouth = document.querySelector('.bot-mouth');
                if(mouth) mouth.classList.remove('talking');
            };
            window.speechSynthesis.speak(utterance);
        } else {
            alert("Speech engine not supported on this browser.");
        }
    };
    </script>
    """,
    unsafe_allow_html=True
)

def render_robot_svg(talking):
    mouth_class = "bot-mouth talking" if talking else "bot-mouth"
    raw = (
        f'<div class="bot-stage"><svg class="bot-float" viewBox="0 0 150 150" xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        '<linearGradient id="bG" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#4f46e5"/></linearGradient>'
        '<linearGradient id="hG" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#c4b5fd"/><stop offset="100%" stop-color="#818cf8"/></linearGradient>'
        '<filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '</defs>'
        '<line x1="75" y1="20" x2="75" y2="6" stroke="#818cf8" stroke-width="2.5" filter="url(#glow)"/>'
        '<circle cx="75" cy="5" r="6" fill="#a78bfa" filter="url(#glow)"/>'
        '<rect x="32" y="80" width="10" height="34" rx="5" fill="url(#bG)" opacity="0.9"/>'
        '<rect x="108" y="80" width="10" height="34" rx="5" fill="url(#bG)" opacity="0.9"/>'
        '<rect x="45" y="78" width="60" height="55" rx="18" fill="url(#bG)"/>'
        '<circle cx="75" cy="105" r="7" fill="#e0e7ff" opacity="0.8"/>'
        '<rect x="35" y="20" width="80" height="62" rx="22" fill="url(#hG)"/>'
        '<ellipse cx="58" cy="50" rx="8" ry="10" fill="#0f172a"/>'
        '<ellipse cx="92" cy="50" rx="8" ry="10" fill="#0f172a"/>'
        '<circle cx="61" cy="47" r="2.8" fill="#fff"/>'
        '<circle cx="95" cy="47" r="2.8" fill="#fff"/>'
        f'<rect class="{mouth_class}" x="60" y="64" width="30" height="7" rx="3.5" fill="#1e1b4b"/>'
        '</svg></div>'
    )
    return raw

def run_chip_query(q):
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role":"user","content":q,"time":now})
    reply, score = get_bot_response(q)
    st.session_state.messages.append({"role":"assistant","content":reply,"time":datetime.now().strftime("%I:%M %p")})
    st.session_state.latest_reply = reply
    st.session_state.bot_talking = True
    st.rerun()

# ── TOP BAR ──
c1, c2, c3 = st.columns([6, 1.4, 1.4])
with c1:
    st.markdown('<h1 style="color:#e2e8f0;font-size:24px;margin:0;">🤖 CodeAlpha FAQ Assistant <span style="font-size:12px;color:#4b5563;font-weight:400;">| Pro 3D Edition</span></h1>', unsafe_allow_html=True)
with c2:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.messages = [defaults["messages"][0]]
        st.session_state.latest_reply = defaults["latest_reply"]
        st.session_state.bot_talking = False
        st.rerun()
with c3:
    transcript = "\n".join([f"[{m['time']}] {m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    st.download_button("⬇ Export", data=transcript, file_name="faq_transcript.txt", use_container_width=True)

# ── CHAT ENGINE DISPLAY ──
st.markdown('<div class="matrix-wrapper">', unsafe_allow_html=True)

robot_markup = render_robot_svg(st.session_state.bot_talking)
chat_html = ""
for msg in st.session_state.messages:
    chat_html += (f'<div class="bubble-row {msg["role"]}">'
                  f'<div class="bubble-node {msg["role"]}">{msg["content"]}</div>'
                  f'<div class="bubble-meta">{msg["time"]}</div></div>')

box_html = (
    f'<div class="chat-terminal-3d">{robot_markup}'
    f'<h2 style="color:#e2e8f0;margin:0;font-weight:800;font-size:20px;">AI Assistant Engine</h2>'
    f'<p style="color:#7c3aed;font-size:10px;font-weight:700;letter-spacing:0.6px;margin-bottom:12px;text-transform:uppercase;">CodeAlpha &#11041; Vector Workspace</p>'
    f'<div class="chat-scroller">{chat_html}</div></div>'
)
st.markdown(box_html, unsafe_allow_html=True)

# ── FORM INPUT ──
with st.form(key="chat_form", clear_on_submit=True):
    fc = st.columns([5,1])
    with fc[0]: user_input = st.text_input("q", placeholder="Type your question here...", label_visibility="collapsed")
    with fc[1]: sent = st.form_submit_button("⚡ Send", use_container_width=True)
    if sent and user_input.strip():
        now = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role":"user","content":user_input,"time":now})
        reply, conf = get_bot_response(user_input)
        st.session_state.messages.append({"role":"assistant","content":reply,"time":datetime.now().strftime("%I:%M %p")})
        st.session_state.latest_reply = reply
        st.session_state.bot_talking = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ── FAQ CARDS GRID ──
st.write("") 
for i in range(0, len(faq_questions), 4):
    row = faq_questions[i:i+4]
    cols = st.columns(4)
    for j, q in enumerate(row):
        with cols[j]:
            if st.button(q, key=f"btn_chip_{i+j}", use_container_width=True):
                run_chip_query(q)