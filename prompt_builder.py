import streamlit as st
import json

# Page Config
st.set_page_config(
    page_title="Detailed Image Prompt Builder",
    page_icon="🎨",
    layout="wide"
)

# --- Session State Management ---
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

def load_example():
    """Loads the Nano Banana example into session state."""
    st.session_state.form_data = {
        # Subject
        "subj_type": "woman",
        "subj_pose": "Mirror selfie, holding a product bottle in the left hand and a smartphone in the right hand to take the picture. She is facing forward, looking slightly off-camera/at her reflection.",
        "subj_expr": "Neutral/soft gaze, slightly pouting lips.",
        "subj_age": "Young adult (appearance)",
        
        # Face Preservation (Default Off)
        "face_preserve": False,
        "face_preserve_desc": "The girl’s facial features, expression, and identity must remain exactly the same as the reference image.",

        # Clothing
        "cloth_top": "Black spaghetti strap crop top/camisole.",
        "cloth_bottom": "Light grey sweatpants or lounge pants, visible waistband.",
        
        # Hair
        "hair_color": "Blonde, with lighter highlights.",
        "hair_style": "Long, wavy/beach waves texture. Features curtain bangs/fringe that frames the forehead and sides of the face.",
        "hair_cond": "Appears voluminous and styled.",
        
        # Face
        "face_feat": "Defined eyebrows, full lips.",
        "face_mkup": "Full coverage, with neutral eyeshadow, winged eyeliner, and a pink/nude lip color. Appears to have blush/contour.",
        
        # Accessories
        "acc_jewel": "None visible.",
        "acc_tech": "Smartphone with a floral/patterned clear case, held in the right hand.",
        "acc_other": "A white bottle of 'OLAPLEX No. 4 Bond Maintenance Shampoo' is held in the left hand.",
        
        # Environment
        "env_set": "Bedroom.",
        "env_furn": "Light wood dresser/cabinet with several drawers. A light wood bed frame and white bedding are visible in the background. A large mirror (used for the selfie) is to the right.",
        "env_decor": "Open wooden shelves on the wall to the left, holding a potted trailing green plant and small white decorative objects (possibly candles/vases). A framed artwork with a light mat and wooden frame is mounted on the wall above the shelf.",
        "env_win": "A window with a white frame and sheer curtain is visible in the background.",
        
        # Lighting
        "light_type": "Natural light, backlighting.",
        "light_src": "Bright sunlight streaming in from the window behind the subject (to the right of her head), creating a strong halo effect and bright highlights. The main subject is relatively well-lit, likely by fill light or reflection.",
        
        # Camera
        "cam_persp": "First-person/selfie, taken from a medium-close distance.",
        "cam_angle": "Slightly high angle, looking down.",
        
        # Style
        "sty_aes": "Lifestyle, beauty, influencer style.",
        "sty_mood": "Casual, soft, bright."
    }

def clear_form():
    """Clears all session state data."""
    st.session_state.form_data = {}

# --- Header & Sidebar ---
st.title("🎨 Detailed Image Prompt Builder")
st.markdown("Create structured, high-fidelity JSON prompts for realistic image generation.")

with st.sidebar:
    st.header("Actions")
    if st.button("🍌 Load 'Nano Banana' Example", type="primary"):
        load_example()
        st.rerun()
    
    if st.button("🗑️ Clear Form"):
        clear_form()
        st.rerun()
        
    st.info("Fill out the tabs on the right to generate your JSON structure.")

# --- Main UI Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["👤 Subject & Look", "👗 Outfit & Accessories", "🏠 Environment & Lighting", "📷 Camera & Style"])

# Helper function to get value safely
def get_val(key, default=""):
    return st.session_state.form_data.get(key, default)

# 1. Subject & Look Tab
with tab1:
    st.subheader("Subject Details")
    col1, col2 = st.columns(2)
    with col1:
        s_type = st.text_input("Type", value=get_val("subj_type"), placeholder="e.g. Woman, Man, Robot")
        s_age = st.text_input("Age", value=get_val("subj_age"), placeholder="e.g. Young adult")
    with col2:
        s_expr = st.text_input("Expression", value=get_val("subj_expr"), placeholder="e.g. Neutral gaze, smiling")
    
    s_pose = st.text_area("Pose", value=get_val("subj_pose"), placeholder="Detailed description of pose...", height=100)

    st.markdown("---")
    
    # --- FACE PRESERVATION LOGIC ---
    col_face_header, col_face_toggle = st.columns([3, 1])
    with col_face_header:
        st.subheader("Face & Hair")
    with col_face_toggle:
        # Toggle for Face Preservation
        preserve_face = st.toggle("📸 Preserve Reference Face", value=get_val("face_preserve", False))

    if preserve_face:
        st.info("ℹ️ **Reference Mode Active:** The JSON will include instructions to lock facial features to a reference image.")
        preserve_desc = st.text_area(
            "Reference Preservation Instruction", 
            value=get_val("face_preserve_desc", "The girl’s facial features, expression, and identity must remain exactly the same as the reference image.")
        )
    else:
        preserve_desc = ""

    col3, col4 = st.columns(2)
    with col3:
        h_color = st.text_input("Hair Color", value=get_val("hair_color"))
        h_cond = st.text_input("Hair Condition", value=get_val("hair_cond"))
        h_style = st.text_area("Hair Style", value=get_val("hair_style"), height=100)
    with col4:
        f_feat = st.text_area("Face Features", value=get_val("face_feat"), height=68)
        f_mkup = st.text_area("Makeup", value=get_val("face_mkup"), height=100)

# 2. Outfit & Accessories Tab
with tab2:
    st.subheader("Clothing")
    c_top = st.text_area("Top", value=get_val("cloth_top"), placeholder="e.g. Black spaghetti strap crop top...")
    c_bot = st.text_area("Bottom", value=get_val("cloth_bottom"), placeholder="e.g. Grey sweatpants...")
    
    st.markdown("---")
    st.subheader("Accessories")
    col1, col2 = st.columns(2)
    with col1:
        a_jewel = st.text_input("Jewelry", value=get_val("acc_jewel"))
        a_tech = st.text_input("Tech", value=get_val("acc_tech"))
    with col2:
        a_other = st.text_area("Other Items", value=get_val("acc_other"), placeholder="Props, items in hand...", height=100)

# 3. Environment & Lighting Tab
with tab3:
    st.subheader("Environment")
    e_set = st.text_input("Setting/Location", value=get_val("env_set"), placeholder="e.g. Bedroom")
    
    col1, col2 = st.columns(2)
    with col1:
        e_furn = st.text_area("Furniture", value=get_val("env_furn"), height=100)
        e_win = st.text_input("Windows", value=get_val("env_win"))
    with col2:
        e_decor = st.text_area("Decor", value=get_val("env_decor"), height=100)
    
    st.markdown("---")
    st.subheader("Lighting")
    l_type = st.text_input("Lighting Type", value=get_val("light_type"), placeholder="e.g. Natural light, Cinematic")
    l_src = st.text_area("Lighting Source/Description", value=get_val("light_src"), placeholder="Direction, shadows, intensity...", height=100)

# 4. Camera & Style Tab
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Camera")
        cam_persp = st.text_input("Perspective", value=get_val("cam_persp"), placeholder="e.g. First-person/selfie")
        cam_angle = st.text_input("Angle", value=get_val("cam_angle"), placeholder="e.g. High angle")
    with col2:
        st.subheader("Style")
        st_aes = st.text_input("Aesthetic", value=get_val("sty_aes"), placeholder="e.g. Lifestyle, Cyberpunk")
        st_mood = st.text_input("Mood", value=get_val("sty_mood"), placeholder="e.g. Casual, Dark")

# --- CONSTRUCT JSON ---

# Base Subject Data
subject_data = {
    "type": s_type,
    "pose": s_pose,
    "expression": s_expr,
    "age": s_age
}

# Base Face Data
face_data = {
    "features": f_feat,
    "makeup": f_mkup
}

# If Preservation is toggled on, inject specific structure into Face Data
if preserve_face:
    face_data["preservation"] = {
        "preserve_original": True,
        "reference_match": True,
        "description": preserve_desc
    }

final_json = {
  "subject": subject_data,
  "clothing": {
    "top": c_top,
    "bottom": c_bot
  },
  "hair": {
    "color": h_color,
    "style": h_style,
    "condition": h_cond
  },
  "face": face_data,
  "accessories": {
    "jewelry": a_jewel,
    "tech": a_tech,
    "other": a_other
  },
  "environment": {
    "setting": e_set,
    "furniture": e_furn,
    "decor": e_decor,
    "windows": e_win
  },
  "lighting": {
    "type": l_type,
    "source": l_src
  },
  "camera": {
    "perspective": cam_persp,
    "angle": cam_angle
  },
  "style": {
    "aesthetic": st_aes,
    "mood": st_mood
  }
}

# --- Output Section ---
st.markdown("---")
st.header("📤 Generated JSON Prompt")

col_json, col_action = st.columns([3, 1])

with col_json:
    st.json(final_json, expanded=False)

with col_action:
    st.success("JSON Ready!")
    json_string = json.dumps(final_json, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_string,
        file_name="prompt_with_face_preservation.json",
        mime="application/json"
    )
    
    st.text_area("Raw Text (Copy manually)", value=json_string, height=300)