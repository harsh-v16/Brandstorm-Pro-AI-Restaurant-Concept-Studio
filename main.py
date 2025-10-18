# =====================================================
# Project: Brandstorm Pro — Restaurant Concept Studio
# Author: Harsh Chaudhary
# Description:
#   A Streamlit web app that helps users brainstorm
#   creative restaurant concepts using GPT-4. It
#   generates names, slogans, logos, interiors,
#   menus, and social copy in one click.
# =====================================================

import os
import base64
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

st.set_page_config(page_title="Brandstorm Pro", page_icon="🚀", layout="wide")

# -----------------------------------------------------
# Load environment variables from .env (for OpenAI key)
# -----------------------------------------------------
load_dotenv()

# -----------------------------------------------------
# BACKGROUND IMAGE SETUP
# -----------------------------------------------------
def set_background(image_file):
    """
    Sets a full-page background image for the Streamlit app.
    The image is encoded in base64 to embed directly in CSS.
    """
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply custom background
set_background("IMG_9773.JPG")

# -----------------------------------------------------
# GLOBAL STYLE SETTINGS (text, buttons, layout)
# -----------------------------------------------------
st.markdown("""
<style>
/* Global font and base color setup */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
    color: #ffffff !important;
}

/* Sidebar customization */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Headings style */
h1, h2, h3, h4, h5, h6, label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Paragraph and small text color */
p, span, small {
    color: #cccccc !important;
}

/* Dropdown / Select styling */
div[data-baseweb="select"] {
    background: rgba(0, 0, 0, 0.4) !important;
    border-radius: 10px !important;
    border: 1px solid #00B4D8 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    transition: all 0.3s ease-in-out;
}
div[data-baseweb="select"]:hover {
    border: 1px solid #48CAE4 !important;
}

/* Dropdown inner text color */
div[data-baseweb="select"] > div {
    background: transparent !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] input {
    background: transparent !important;
    color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Dropdown arrow color */
div[data-baseweb="select"] svg {
    color: #ffffff !important;
}

/* Dropdown popup (popover) styling */
div[data-baseweb="popover"] {
    background: linear-gradient(90deg, #004AAD, #00B4D8) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}
div[data-baseweb="popover"] li {
    color: #ffffff !important;
    background: transparent !important;
}
div[data-baseweb="popover"] li:hover {
    background: rgba(255,255,255,0.15) !important;
}

/* Text input field styling */
.stTextInput>div>div>input {
    background: rgba(255,255,255,0.15);
    color: #ffffff !important;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.3);
}
.stTextInput>label {
    color: #ffffff !important;
}

/* Main glowing gradient button */
.stButton>button {
    background: linear-gradient(90deg, #004AAD, #00B4D8);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 10px 22px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: all 0.25s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00B4D8, #48CAE4);
    transform: translateY(-2px);
}

/* Example buttons (under cuisine examples) */
div[data-testid="stHorizontalBlock"] button {
    background: linear-gradient(90deg, #004AAD, #00B4D8);
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600;
    border: none !important;
    transition: all 0.2s ease-in-out;
}
div[data-testid="stHorizontalBlock"] button:hover {
    background: linear-gradient(90deg, #00B4D8, #48CAE4);
    transform: translateY(-2px);
}

/* Slider bar styling */
div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #004AAD, #00B4D8) !important;
}

/* Glass panel effect for main content */
section.main > div {
    background: rgba(0, 0, 0, 0.4);
    border-radius: 12px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# STREAMLIT APP HEADER
# -----------------------------------------------------

st.title("🚀 Brandstorm Pro — Restaurant Concept Studio")
st.caption("Generate names, taglines, logo ideas, menus, design concepts and social copy — all in one click.")
st.divider()

# -----------------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------------
with st.sidebar:
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("**Model:** GPT-4 🔥")  # Fixed model display
    temperature = st.sidebar.slider("Creativity Level", 0.2, 1.0, 0.7)
    st.sidebar.info("Higher creativity gives more unique ideas.")
    st.sidebar.markdown("💡 Tip: Enter any cuisine or theme to explore restaurant concepts.")
    st.divider()
    st.info("💡 Tip: Enter any cuisine, like 'Spicy Mexican' or 'Vegan Italian'")

# -----------------------------------------------------
# MAIN USER INPUT
# -----------------------------------------------------
EXAMPLES = [
    "Indian street food fusion",
    "Luxury seafood",
    "Vegan comfort food",
    "Spicy Mexican",
    "Sustainable farm-to-table"
]

st.markdown("### 🍽️ Describe your restaurant concept")
cuisine = st.text_input("Cuisine / Theme", placeholder="e.g., 'Vegan Italian' or 'Fusion Street Food'")

# Optional user input for favorite dishes
favorite_dishes = st.text_input("Your Favorite Dishes (optional)", placeholder="e.g., Butter Chicken, Ramen, Tacos")

# Example quick-pick buttons
st.markdown("### 🍴 Or pick an example")
cols = st.columns(5)
for idx, example in enumerate(EXAMPLES):
    with cols[idx]:
        if st.button(example, use_container_width=True):
            cuisine = example
            st.session_state._prefill = example

st.divider()

# User output preferences
st.markdown("### 🎯 Output Options")
want_multiple_taglines = st.checkbox("Produce 3 tagline variations", value=True)
want_logo_description = st.checkbox("Generate Logo Idea + Color Palette", value=True)
want_interior = st.checkbox("Interior Design Suggestion", value=True)
want_social = st.checkbox("Social Media Copy (IG bio + 1 post)", value=True)

# -----------------------------------------------------
# GPT-4 CONCEPT GENERATION FUNCTION
# -----------------------------------------------------
def generate_concept(cuisine, model_selected, temperature):
    """
    Uses GPT-4 (via LangChain) to generate a complete restaurant concept.
    It produces a name, slogan, taglines, logo idea, interior style,
    menu items, and social media copy.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    # Define multiple LLM chains for modular outputs
    name_prompt = ChatPromptTemplate.from_template("Suggest one creative and short restaurant name for {cuisine}.")
    name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key="restaurant_name")

    slogan_prompt = ChatPromptTemplate.from_template("Create a catchy one-line slogan for the restaurant '{restaurant_name}'.")
    slogan_chain = LLMChain(llm=llm, prompt=slogan_prompt, output_key="slogan")

    tagline_prompt = ChatPromptTemplate.from_template("Generate 3 short tagline options for '{restaurant_name}'. Separate them with commas.")
    tagline_chain = LLMChain(llm=llm, prompt=tagline_prompt, output_key="taglines")

    logo_prompt = ChatPromptTemplate.from_template("Describe a creative logo idea and 3-color palette for '{restaurant_name}'.")
    logo_chain = LLMChain(llm=llm, prompt=logo_prompt, output_key="logo_idea")

    interior_prompt = ChatPromptTemplate.from_template("Describe the interior design style for '{restaurant_name}' in 3 bullet points.")
    interior_chain = LLMChain(llm=llm, prompt=interior_prompt, output_key="interior_design")

    menu_prompt = ChatPromptTemplate.from_template("List 5 signature dishes for '{restaurant_name}' with short descriptions.")
    menu_chain = LLMChain(llm=llm, prompt=menu_prompt, output_key="menu_items")

    social_prompt = ChatPromptTemplate.from_template("Write a short Instagram bio and one promotional post for '{restaurant_name}'.")
    social_chain = LLMChain(llm=llm, prompt=social_prompt, output_key="social_copy")

    # Combine all LLMChains sequentially
    overall_chain = SequentialChain(
        chains=[name_chain, slogan_chain, tagline_chain, logo_chain, interior_chain, menu_chain, social_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "slogan", "taglines", "logo_idea", "interior_design", "menu_items", "social_copy"],
        verbose=False,
    )

    # Execute the chain
    response = overall_chain.invoke({"cuisine": cuisine})
    return response

# -----------------------------------------------------
# GENERATE BUTTON WITH ANIMATED GLOW
# -----------------------------------------------------
st.markdown("### ✨ Generate Your Concept")

# Custom CSS for subtle glow animation
st.markdown("""
<style>
@keyframes glowPulse {
  0% { box-shadow: 0 0 8px rgba(0,180,216,0.4), 0 0 15px rgba(0,180,216,0.3); }
  50% { box-shadow: 0 0 20px rgba(0,180,216,0.7), 0 0 40px rgba(0,180,216,0.6); }
  100% { box-shadow: 0 0 8px rgba(0,180,216,0.4), 0 0 15px rgba(0,180,216,0.3); }
}
.stButton>button.generate-btn {
  background: linear-gradient(90deg, #004AAD, #00B4D8);
  color: white;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  padding: 12px 26px;
  animation: glowPulse 2.5s infinite ease-in-out;
  transition: all 0.3s ease-in-out;
}
.stButton>button.generate-btn:hover {
  background: linear-gradient(90deg, #00B4D8, #48CAE4);
  transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# Button component
generate = st.button(
    "✨ Generate Concept",
    key="generate",
    type="primary",
    help="Click to generate your restaurant brand concept",
)

# -----------------------------------------------------
# EXECUTION: Run AI only when button clicked
# -----------------------------------------------------
if not cuisine:
    st.warning("Please enter or pick a cuisine/theme.")
elif generate:
    with st.spinner("Brainstorming your restaurant concept... 🍳"):
        # Run GPT-4 chain and display structured outputs
        result = generate_concept(cuisine=cuisine, model_selected="gpt-4", temperature=temperature)

        st.divider()
        st.subheader(f"AI Results for *{cuisine.title()}*")

        # --- Restaurant Name + Slogan ---
        st.markdown(f"### 🏷️ **{result['restaurant_name'].strip()}**")
        st.markdown(f"**💬 Slogan:** _{result['slogan'].strip()}_")

        # --- Taglines ---
        if want_multiple_taglines:
            taglines = [t.strip() for t in result['taglines'].split(",") if t.strip()]
            st.markdown("**🪄 Tagline Variations:**")
            for t in taglines:
                st.write(f"- {t}")

        # --- Logo Idea ---
        if want_logo_description:
            st.markdown("### 🎨 Logo & Color Palette")
            st.write(result["logo_idea"])

        # --- Interior Design ---
        if want_interior:
            st.markdown("### 🏠 Interior Design Style")
            st.write(result["interior_design"])

        # --- Menu Items ---
            st.markdown("### 🍽️ Signature Dishes")
            st.write(result["menu_items"])

        # --- Social Copy ---
        if want_social:
            st.markdown("### 📱 Social Media Copy")
            st.write(result["social_copy"])

        st.success("✨ Concept generated successfully!")

# =====================================================
# End of Script
# =====================================================






