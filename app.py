import os, sys
import time
import pandas as pd
import streamlit as st

# Ensure src is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="TicketIQ — AI Support Classifier",
    page_icon="🎫",
    layout="wide"
)

# Improved CSS for better box and table alignment
st.markdown("""
<style>
body { background: #0b1326; color: #dae2fd; }
.card, .input-panel, .result-panel {
    background: #171f33;
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(60,72,100,0.10);
}
.rcard {
    background: #222a3d;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
}
.rcard-label { font-size: 1rem; color: #a1b1ff; font-weight: 600; }
.rcard-value { font-size: 1.4rem; font-weight: bold; }
.badge-High { color: #ef4444; font-weight: bold; }
.badge-Medium { color: #f59e0b; font-weight: bold; }
.badge-Low { color: #22c55e; font-weight: bold; }
.route-box {
    background: #2d3449;
    padding: 1rem 1.5rem;
    border-left: 4px solid #b8c3ff;
    border-radius: 10px;
    margin-top: 1rem;
}
/* Table improvements */
table { width: 100%; border-collapse: separate; border-spacing: 0; }
th, td { text-align: left; padding: 0.85rem 1rem; }
th { background: #1a2133; color: #b8c3ff; font-size: 1rem; font-weight: 700; border-bottom: 2px solid #222a3d; }
td { background: #171f33; color: #dae2fd; font-size: 1rem; border-bottom: 1px solid #222a3d; }
tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #232a44; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    from src.predict import TicketClassifier
    return TicketClassifier(models_dir="models/")

try:
    clf = load_model()
    model_ok = True
except Exception as e:
    st.error(f"Model load error: {e}")
    model_ok = False

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None


# (Removed duplicate dashboard section above. Only the improved dashboard below remains.)


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
CATEGORY_ICON = {
    "Billing Issue":             "💳",
    "Technical Support":         "⚙️",
    "Account Management":        "👤",
    "Product Inquiry":           "📦",
    "Shipping and Delivery":     "🚚",
    "Refund Request":            "💰",
    "Feedback and Suggestions":  "💡",
}
CONF_COLOR = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}

SAMPLES = [
    ("Billing",   "I was charged twice on my credit card this month. Please refund the duplicate payment."),
    ("Tech",      "The mobile app crashes every time I open the settings page. Reinstalling didn't help."),
    ("Account",   "I forgot my password and the reset email never arrives, even after checking spam."),
    ("Shipping",  "My package was marked delivered but I never received it. Tracking shows it at door."),
    ("Inquiry",   "Does your Pro plan support integration with Salesforce CRM?"),
    ("Refund",    "I returned the product 3 weeks ago and the refund still hasn't appeared on my card."),
    ("Feedback",  "The new dashboard layout is confusing. Can you add an option to revert to classic view?"),
]


# ─────────────────────────────────────────────────────────────
#  Load model (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    from src.predict import TicketClassifier
    return TicketClassifier(models_dir="models/")

try:
    clf = load_classifier()
    model_ok = True
except Exception as e:
    model_ok = False


# ─────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None



# Sidebar
with st.sidebar:
        st.markdown('<div class="logo">🎫 TicketIQ AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">V2.4 Analytical Luminary</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav">'
                                '<a class="active">Dashboard</a>'
                                '<a>Classification</a>'
                                '<a>Batch Upload</a>'
                                '<a>Performance</a>'
                                '</div>', unsafe_allow_html=True)
        st.markdown('<hr style="border:1px solid #2d3449; margin:2rem 0;">', unsafe_allow_html=True)
        st.markdown('<a class="nav" style="color:#a1b1ff;">Help Center</a>', unsafe_allow_html=True)
        st.markdown('<a class="nav" style="color:#a1b1ff;">Documentation</a>', unsafe_allow_html=True)

# Topbar
st.markdown('<div class="topbar">'
                        '<div class="title">TicketIQ</div>'
                        '<div class="nav">'
                        '<a class="active">Dashboard</a>'
                        '<a>Classify</a>'
                        '<a>Analytics</a>'
                        '<a>History</a>'
                        '</div>'
                        '</div>', unsafe_allow_html=True)

# Hero Section
st.markdown('<h2 class="gradient-text" style="font-size:2.5rem;font-weight:900;margin-bottom:0.5rem;">🎫 TicketIQ — AI Support Classifier</h2>', unsafe_allow_html=True)
st.markdown('<div style="display:flex;gap:1rem;margin-bottom:2.5rem;">'
                        '<span class="font-label text-xs tracking-widest uppercase px-3 py-1 bg-surface-container-highest text-primary rounded-full" style="background:#222a3d;color:#b8c3ff;">7 Categories</span>'
                        '<span class="font-label text-xs tracking-widest uppercase px-3 py-1 bg-surface-container-highest text-primary rounded-full" style="background:#222a3d;color:#b8c3ff;">Logistic Regression</span>'
                        '<span class="font-label text-xs tracking-widest uppercase px-3 py-1 bg-surface-container-highest text-primary rounded-full" style="background:#222a3d;color:#b8c3ff;">TF-IDF Features</span>'
                        '</div>', unsafe_allow_html=True)



# Main grid layout

# Render only the intended content boxes at the top, without extra columns or containers
st.markdown('<div style="display:flex;gap:2rem;margin-bottom:2rem;">'
    '<div class="card" style="flex:7;background:#171f33;">'
        '<div style="color:#b8c3ff;font-size:1.15rem;font-weight:600;">Welcome to TicketIQ!</div>'
        '<div style="color:#dae2fd;font-size:1rem;">Classify support tickets by category and urgency using AI. Enter a ticket below to get started.</div>'
    '</div>'
    '<div class="card" style="flex:5;background:#171f33;">'
        '<div style="color:#b8c3ff;font-size:1.15rem;font-weight:600;">How it works</div>'
        '<div style="color:#dae2fd;font-size:1rem;">Paste or type a customer support ticket. The model predicts its category (e.g., Billing, Technical) and assigns a priority level.</div>'
    '</div>'
'</div>', unsafe_allow_html=True)

# Restore columns for main layout
left, right = st.columns([7, 5], gap="large")


# LEFT COLUMN: Input & Model Info
with left:
    st.markdown('<div class="card input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="font-size:1.2rem;font-weight:700;color:#b8c3ff;margin-bottom:1rem;">Ticket Description</div>', unsafe_allow_html=True)

    # Initialize ticket_box if not present
    if "ticket_box" not in st.session_state:
        st.session_state["ticket_box"] = ""

    def set_sample(text):
        st.session_state["ticket_box"] = text
        st.session_state.result = None

    ticket_text = st.text_area(
        label="Ticket description",
        label_visibility="collapsed",
        placeholder="Paste the customer support ticket text here... e.g., 'I am unable to login to my account after changing my password yesterday...'",
        height=140,
        key="ticket_box",
    )
    st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown('<p class="font-label" style="font-size:0.85rem;letter-spacing:1.5px;text-transform:uppercase;color:#a1b1ff;margin-bottom:0.5rem;">Quick Samples</p>', unsafe_allow_html=True)
    sample_cols = st.columns(6)
    for i, (label, sample_text) in enumerate(SAMPLES):
        with sample_cols[i % 6]:
            st.button(label, key=f"smpl_{label}", on_click=set_sample, args=(sample_text,))
    st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        classify = st.button("🚀 Classify Ticket", key="classify_btn", disabled=not model_ok)
    with c2:
        if st.button("🧹 Clear", key="clear_btn"):
            st.session_state["ticket_box"] = ""
            st.session_state.result = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Info Box
    st.markdown('<div class="card" style="background:#171f33;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="color:#a1b1ff;">Model Architecture</div>', unsafe_allow_html=True)
    mcols = st.columns(3)
    mcols[0].markdown('<div class="stat-card">Logistic Regression<br><span style="font-size:0.8rem;font-weight:400;">Algorithm</span></div>', unsafe_allow_html=True)
    mcols[1].markdown('<div class="stat-card">TF-IDF 1-3 Grams<br><span style="font-size:0.8rem;font-weight:400;">Vectorization</span></div>', unsafe_allow_html=True)
    mcols[2].markdown('<div class="stat-card">8,500<br><span style="font-size:0.8rem;font-weight:400;">Training Set</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



# RIGHT COLUMN: Results & History
with right:
    st.markdown('<div class="card result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Classification Results</div>', unsafe_allow_html=True)
    # Run classification when button clicked
    if classify:
        if ticket_text.strip():
            with st.spinner(""):
                time.sleep(0.35)
                st.session_state.result = clf.predict(ticket_text)
                st.session_state.history.insert(0, st.session_state.result)
        else:
            st.warning("Please enter a ticket description first.")
    res = st.session_state.result
    if res:
        cat    = res["category"]
        pri    = res["priority"]
        c_conf = res["category_confidence"]
        p_conf = res["priority_confidence"]
        icon   = CATEGORY_ICON.get(cat, "🏷️")
        p_col  = CONF_COLOR.get(pri, "#f59e0b")
        c_pct  = f"{c_conf*100:.1f}"
        p_pct  = f"{p_conf*100:.1f}"
        # Category card
        st.markdown(f"""
        <div class="rcard">
            <div class="rcard-label">Predicted Category</div>
            <div class="rcard-value" style="font-size:2.2rem;font-weight:900;">{icon}&nbsp;{cat}</div>
            <div class="rcard-conf-track" style="margin-top:0.7rem;">
                <div class="rcard-conf-fill" style="width:{c_pct}%;background:linear-gradient(90deg,#b8c3ff,#2d5bff);height:12px;"></div>
            </div>
            <div class="rcard-conf-text" style="font-size:1.1rem;color:#b8c3ff;">{c_pct}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
        # Priority card
        st.markdown(f"""
        <div class="rcard">
            <div class="rcard-label">Assigned Priority</div>
            <div class="rcard-value" style="gap:0.75rem;font-size:1.5rem;">
                <span class="badge badge-{pri}">{pri}</span>
            </div>
            <div class="rcard-conf-track" style="margin-top:0.7rem;">
                <div class="rcard-conf-fill" style="width:{p_pct}%;background:{p_col};height:12px;"></div>
            </div>
            <div class="rcard-conf-text" style="font-size:1.1rem;color:#b8c3ff;">{p_pct}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
        # Routing recommendation
        st.markdown(f"""
        <div class="route-box" style="background:#2d3449;border-left:4px solid #b8c3ff;color:#a1b1ff;">
            <span class="material-symbols-outlined" style="vertical-align:middle;font-size:1.3rem;">route</span>
            <strong style="color:#b8c3ff">Routing Recommendation</strong><br>
            <span style="color:#dae2fd">{res['recommendation']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎫</div>
            <div class="empty-title">Ready to classify</div>
            <div class="empty-sub">Enter a ticket and click Classify Ticket</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)




# History Section
if st.session_state.history:
    st.markdown('<div class="card" style="background:#171f33;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="font-size:1.1rem;font-weight:700;color:#b8c3ff;margin-bottom:1rem;">🕐 Classification History</div>', unsafe_allow_html=True)
    h1, h2 = st.columns([5, 1])
    with h2:
        if st.button("Clear all", key="clr_hist"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    st.markdown('<div style="overflow-x:auto;">', unsafe_allow_html=True)
    # Table header
    st.markdown('<table>'
                '<thead><tr>'
                '<th>#</th>'
                '<th>Ticket Snippet</th>'
                '<th>Category</th>'
                '<th>Confidence</th>'
                '<th>Priority</th>'
                '</tr></thead><tbody>', unsafe_allow_html=True)
    for i, r in enumerate(st.session_state.history[:8]):
        icon   = CATEGORY_ICON.get(r["category"], "🏷")
        badge  = f'<span class="badge badge-{r["priority"]}">{r["priority"]}</span>'
        snippet = r["original_text"][:85] + ("…" if len(r["original_text"]) > 85 else "")
        conf = f'{r["category_confidence"]*100:.1f}%'
        st.markdown(f"<tr>"
                    f"<td>{i+1}</td>"
                    f"<td>{snippet}</td>"
                    f"<td>{icon} {r['category']}</td>"
                    f"<td>{conf}</td>"
                    f"<td>{badge}</td>"
                    f"</tr>", unsafe_allow_html=True)
    st.markdown('</tbody></table></div>', unsafe_allow_html=True)
    df_csv = pd.DataFrame([{
        "Text": r["original_text"],
        "Category": r["category"],
        "Category Confidence": f"{r['category_confidence']:.2%}",
        "Priority": r["priority"],
        "Priority Confidence": f"{r['priority_confidence']:.2%}",
        "Recommendation": r["recommendation"],
    } for r in st.session_state.history])
    st.download_button(
        "⬇️  Download History as CSV",
        data=df_csv.to_csv(index=False),
        file_name="ticket_classifications.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)


# Model error banner
if not model_ok:
    st.error("⚠️ Models not found — run `python main.py` first to train and save the models.")
