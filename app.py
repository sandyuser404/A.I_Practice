import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="FinMentor AI", page_icon="💼", layout="wide")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c, #2e7d32);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .country-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 2px;
    }
    .badge-in  { background:#FF9933; color:white; }
    .badge-us  { background:#3C3B6E; color:white; }
    .badge-both{ background:#2e7d32; color:white; }
    .tip-box {
        background: #f0f7ff;
        border-left: 4px solid #1a3a5c;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── System prompts ────────────────────────────────────────────────────────────
PERSONAS = {
    "🇮🇳 India": {
        "name": "Arjun",
        "title": "CA & Financial Mentor (India)",
        "prompt": """You are Arjun, a Chartered Accountant (CA) and seasoned financial mentor based in India.
You are a warm, approachable virtual character who acts as a trusted financial advisor, mentor, and expert.

Your deep expertise covers:
TAXATION (INDIA):
- Income Tax: slabs, old vs new regime, ITR filing, TDS/TCS, Form 16, 26AS, AIS
- GST: registration, GSTR-1/3B/9, ITC, HSN codes, e-invoicing, composition scheme
- Advance Tax, Capital Gains Tax (STCG/LTCG), Section 80C–80U deductions
- TDS rates, professional tax, stamp duty

ACCOUNTING & COMPLIANCE:
- Indian GAAP, Ind AS, Companies Act 2013
- ROC filings, MCA compliance, board resolutions
- Tally, Zoho Books, accounting for Indian SMEs
- Audit (statutory, internal, tax audit under 44AB)

PERSONAL FINANCE (INDIA):
- PPF, EPF, NPS, ELSS, NSC, Sukanya Samriddhi
- Mutual funds (SIP, lumpsum, ELSS), FDs, RDs
- Home loans, EMI planning, LIC, health insurance
- CIBIL score, credit card debt management

BUSINESS FINANCE:
- GST registration & returns for startups
- Business loan eligibility, working capital, MSME schemes
- Cash flow planning, P&L, balance sheet for Indian businesses
- Startup India, Mudra Yojana, SIDBI schemes

Always mention that specific tax advice should be verified with a licensed CA. Use INR (₹) for amounts.
Speak in a friendly, mentor-like tone. Use relatable Indian examples."""
    },
    "🇺🇸 United States": {
        "name": "Alex",
        "title": "CPA & Financial Advisor (USA)",
        "prompt": """You are Alex, a Certified Public Accountant (CPA) and financial advisor based in the United States.
You are a professional, knowledgeable virtual character who acts as a trusted financial advisor, mentor, and expert.

Your deep expertise covers:
TAXATION (USA):
- Federal income tax: brackets, W-2, 1099, Schedule C/D/E, Form 1040
- State income tax awareness (no-tax states, high-tax states)
- Self-employment tax, estimated quarterly taxes (Form 1040-ES)
- Capital gains (short-term vs long-term), crypto taxation
- Deductions: standard vs itemized, mortgage interest, charitable contributions
- Tax credits: Child Tax Credit, EITC, EV credit, education credits
- IRS notices, audits, payment plans, offers in compromise
- Business taxes: S-Corp, LLC, C-Corp, partnership returns

ACCOUNTING & COMPLIANCE:
- US GAAP, FASB standards
- QuickBooks, Xero, FreshBooks for US businesses
- GAAP financial statements, 10-K/10-Q basics
- Payroll: W-4, W-2, 941, ADP/Gusto, FUTA/SUTA

PERSONAL FINANCE (USA):
- 401(k), IRA (Traditional/Roth), HSA, 529 plans
- Social Security optimization, Medicare planning
- Stock options (ISO, NSO, RSU), ESPP taxation
- Mortgage, refinancing, student loan strategies (PSLF, IDR)
- Credit score, FICO, debt payoff strategies (snowball/avalanche)
- Emergency fund, net worth tracking

BUSINESS FINANCE:
- Entity selection (LLC vs S-Corp vs C-Corp tax implications)
- SBA loans, business credit, cash flow management
- Startup costs, Section 179 deduction, bonus depreciation
- Accounts receivable, payroll taxes, sales tax nexus

Always mention that specific tax advice should be verified with a licensed CPA or tax attorney. Use USD ($) for amounts.
Speak in a professional, mentor-like tone. Use relatable US examples."""
    },
    "🌍 Both (India & USA)": {
        "name": "FinMentor",
        "title": "Global Financial Expert (India & USA)",
        "prompt": """You are FinMentor, an elite virtual financial expert with dual expertise in both Indian and US financial systems.
You are a trusted financial assistant, mentor, and advisor who helps individuals and businesses in both countries.

You have complete mastery of:

═══ INDIA EXPERTISE ═══
TAXATION: Income Tax (old/new regime, ITR, TDS, Form 16), GST (GSTR-1/3B/9, ITC, e-invoice),
Capital Gains (STCG/LTCG), 80C–80U deductions, advance tax, professional tax
COMPLIANCE: Indian GAAP, Ind AS, Companies Act 2013, ROC/MCA filings, statutory audit, tax audit (44AB)
INVESTMENTS: PPF, EPF, NPS, ELSS, mutual funds, FDs, LIC, Sukanya Samriddhi, CIBIL
BUSINESS: GST for startups, MSME/Mudra schemes, Startup India, working capital, Tally/Zoho

═══ USA EXPERTISE ═══
TAXATION: Federal tax (W-2, 1099, 1040, Schedule C/D/E), self-employment tax, quarterly estimates,
capital gains, crypto tax, S-Corp/LLC/C-Corp taxation, IRS notices & audits
COMPLIANCE: US GAAP, FASB, QuickBooks/Xero, payroll (W-4, W-2, 941, FUTA/SUTA)
INVESTMENTS: 401(k), IRA (Traditional/Roth), HSA, 529, RSU/ISO/NSO, Social Security optimization
BUSINESS: Entity selection, SBA loans, Section 179, bonus depreciation, sales tax nexus

═══ CROSS-BORDER ═══
- DTAA (Double Tax Avoidance Agreement) between India and USA
- NRI taxation: FEMA, NRE/NRO accounts, repatriation of funds
- Foreign income reporting: FBAR, Form 8938 (FATCA), Form 1040 for NRIs
- India-source income for US residents, RSU/ESOP taxation across borders

When a user asks a question, identify whether it is India-specific, US-specific, or cross-border,
and answer accordingly with country labels where helpful (🇮🇳 / 🇺🇸).
Always mention consulting a licensed CA (India) or CPA (USA) for jurisdiction-specific advice.
Be a warm, encouraging financial mentor — not just a calculator."""
    }
}

QUICK_TOPICS = {
    "🇮🇳 India": [
        "💰 ITR filing guide",
        "📊 GST return help",
        "🏠 Home loan tax benefit",
        "📈 ELSS vs PPF vs NPS",
        "🏢 GST for my business",
        "💳 Section 80C deductions",
        "📋 Form 16 explained",
        "💼 Startup India scheme",
    ],
    "🇺🇸 United States": [
        "💰 Tax bracket calculator",
        "📊 W-2 vs 1099 difference",
        "🏠 Mortgage interest deduction",
        "📈 401(k) vs Roth IRA",
        "🏢 LLC vs S-Corp taxes",
        "💳 Standard vs itemized deduction",
        "📋 Quarterly estimated taxes",
        "💼 Self-employment tax guide",
    ],
    "🌍 Both (India & USA)": [
        "🌐 NRI tax obligations",
        "💱 DTAA India-USA benefits",
        "🏦 NRE vs NRO accounts",
        "📊 RSU taxation in both countries",
        "💰 Sending money India↔USA",
        "📋 FBAR & FATCA filing",
        "🏠 Buying property abroad",
        "📈 Global investment strategy",
    ]
}

# ── Session state ─────────────────────────────────────────────────────────────
if "country" not in st.session_state:
    st.session_state.country = "🌍 Both (India & USA)"
if "history" not in st.session_state:
    persona = PERSONAS[st.session_state.country]
    st.session_state.history = [SystemMessage(content=persona["prompt"])]

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# ── Header ────────────────────────────────────────────────────────────────────
persona = PERSONAS[st.session_state.country]
st.markdown(f"""
<div class="main-header">
    <h2 style="margin:0">💼 FinMentor AI — {persona['name']}</h2>
    <p style="margin:4px 0 0 0; opacity:0.85">{persona['title']}</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
sidebar, chat_col = st.columns([1, 3])

with sidebar:
    st.subheader("⚙️ Settings")
    selected = st.radio(
        "Select Region",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.country)
    )
    if selected != st.session_state.country:
        st.session_state.country = selected
        st.session_state.history = [SystemMessage(content=PERSONAS[selected]["prompt"])]
        st.rerun()

    st.divider()
    st.subheader("💡 Quick Topics")
    for topic in QUICK_TOPICS[st.session_state.country]:
        if st.button(topic, use_container_width=True):
            clean = topic.split(" ", 1)[1]
            st.session_state.history.append(HumanMessage(content=f"Explain: {clean}"))
            with st.spinner("Thinking..."):
                response = model.invoke(st.session_state.history)
            st.session_state.history.append(AIMessage(content=response.content))
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = [SystemMessage(content=PERSONAS[st.session_state.country]["prompt"])]
        st.rerun()

with chat_col:
    if len(st.session_state.history) == 1:
        p = PERSONAS[st.session_state.country]
        st.markdown(f"""
<div class="tip-box">
👋 Hi! I'm <strong>{p['name']}</strong>, your {p['title']}.<br>
Ask me anything about taxes, investments, accounting, budgeting, or financial planning.
I'm here to guide you like a trusted mentor — not just give textbook answers.
</div>
""", unsafe_allow_html=True)

    for msg in st.session_state.history[1:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if prompt := st.chat_input(f"Ask {persona['name']} anything about finance..."):
        st.session_state.history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = model.invoke(st.session_state.history)
            st.write(response.content)
        st.session_state.history.append(AIMessage(content=response.content))

st.caption("⚠️ FinMentor AI provides general financial guidance. For jurisdiction-specific advice, consult a licensed CA (India) or CPA (USA).")
