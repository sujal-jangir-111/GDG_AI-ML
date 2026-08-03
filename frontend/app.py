
import os

# Override EasyOCR directory to store models inside project folder
LOCAL_OCR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "easyocr_data"))
os.environ["EASYOCR_MODULE_PATH"] = LOCAL_OCR_DIR
os.makedirs(LOCAL_OCR_DIR, exist_ok=True)

import difflib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Medicine Price & Alternative Finder", 
    page_icon="💊", 
    layout="wide"
)

# 2. Global Backend URL & Paths
API_URL = "https://gdg-ai-ml.onrender.com"  # "http://127.0.0.1:8000"

# Local directory for storing EasyOCR models (Bypasses Windows C:\Users Permission Error)
OCR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "ocr_models")
os.makedirs(OCR_MODEL_DIR, exist_ok=True)


# 3. Load Brand Names for OCR Fuzzy Matching
@st.cache_data
def load_brand_names():
    try:
        data_path = os.path.join(os.path.dirname(__file__), '../data/cleaned_medicines.csv')
        df_local = pd.read_csv(data_path, usecols=['brand_name'])
        return df_local['brand_name'].drop_duplicates().tolist()
    except Exception:
        return []

all_brand_names = load_brand_names()


# 4. Cache EasyOCR Reader
@st.cache_resource
def load_ocr_reader():
    try:
        import easyocr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The OCR backend 'easyocr' is not installed. Install it with 'pip install easyocr' "
            "and restart Streamlit."
        ) from exc
    return easyocr.Reader(['en'], gpu=False, model_storage_directory=LOCAL_OCR_DIR)

# 5. Matching OCR Output to Known Brand Names
def match_ocr_to_brand(extracted_lines, brand_list):
    if not extracted_lines or not brand_list:
        return None
        
    full_text_lower = " ".join(extracted_lines).lower()
    sorted_brands = sorted(brand_list, key=len, reverse=True)
    
    # Check if brand name exists as substring inside image text
    for brand in sorted_brands:
        brand_clean = brand.lower()
        core_brand = brand_clean.replace("tablet", "").replace("capsule", "").replace("injection", "").strip()
        
        if len(core_brand) >= 3 and core_brand in full_text_lower:
            return brand

    # Line-by-Line Fuzzy Matching
    for line in extracted_lines:
        line_clean = line.strip()
        if len(line_clean) >= 3:
            matches = difflib.get_close_matches(line_clean, brand_list, n=1, cutoff=0.4)
            if matches:
                return matches[0]

    # Word-by-Word Fuzzy Matching
    for line in extracted_lines:
        for word in line.split():
            if len(word) >= 4:
                matches = difflib.get_close_matches(word, brand_list, n=1, cutoff=0.5)
                if matches:
                    return matches[0]

    return None


# --- SIDEBAR ---
st.sidebar.title("💊 Medicine Finder")
st.sidebar.info("Upload a medicine strip photo or type a brand name to find fair prices and generic alternatives.")

st.sidebar.subheader("💡 Sample Medicines to Try")
sample_med = st.sidebar.selectbox(
    "Click a sample medicine to test:",
    [
        "Select a sample...",
        "Lulifer 1% Cream",
        "Moxind 500mg Capsule",
        "Ero 150mg Tablet",
        "Dial 0.5mg Tablet"
    ]
)

# --- MAIN HEADER ---
st.title("💊 Medicine Price & Generic Alternative Finder")
st.markdown("Find out if you are overpaying for your medicine and discover cheaper alternatives with the exact same active ingredients!")

# --- OPTIONAL OCR SECTION ---
st.subheader("📷 Scan Medicine Strip (Optional OCR)")
uploaded_image = st.file_uploader("Upload a photo of a medicine strip or box:", type=["jpg", "jpeg", "png"])

ocr_detected_name = ""

if uploaded_image is not None:
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Medicine Photo", width=220)
        
    with col_info:
        with st.spinner("Scanning image using EasyOCR..."):
            try:
                reader = load_ocr_reader()
            except ModuleNotFoundError as e:
                st.error(str(e))
                reader = None

            if reader is not None:
                try:
                    img_np = np.array(image)
                    
                    # Extract text lines from image
                    extracted_lines = reader.readtext(img_np, detail=0)
                    
                    if extracted_lines:
                        st.write("📝 **Text Read by EasyOCR:**")
                        st.code("\n".join(extracted_lines))
                        
                        # Run line/word matching
                        ocr_detected_name = match_ocr_to_brand(extracted_lines, all_brand_names)
                        
                        if ocr_detected_name:
                            st.success(f"🎯 **Matched Medicine in Dataset:** `{ocr_detected_name}`")
                        else:
                            st.warning("⚠️ Extracted text, but couldn't match an exact brand name in dataset. Please type it manually below.")
                    else:
                        st.warning("No text detected in image. Please type the medicine name manually below.")
                except Exception as e:
                    st.error(f"OCR Exception: {str(e)}")

# --- SEARCH BAR ---
default_search = ""
if ocr_detected_name:
    default_search = ocr_detected_name
elif sample_med != "Select a sample...":
    default_search = sample_med

medicine_name = st.text_input(
    "Enter Medicine Brand Name:", 
    value=default_search,
    placeholder="e.g., Moxind 500mg Capsule"
)

# --- SEARCH ACTION ---
if st.button("Search Medicine", type="primary"):
    if medicine_name.strip():
        with st.spinner("Searching dataset and calculating fair market price..."):
            try:
                price_res = requests.get(f"{API_URL}/predict-price", params={"medicine_name": medicine_name})
                alt_res = requests.get(f"{API_URL}/alternatives", params={"medicine_name": medicine_name})
                
                if price_res.status_code == 200 and alt_res.status_code == 200:
                    price_data = price_res.json()
                    alt_data = alt_res.json()
                    
                    actual_price = price_data.get("actual_price", 0)
                    fair_price = price_data.get("predicted_fair_price", 0)
                    composition = price_data.get("composition", "N/A")
                    alternatives = alt_data.get("alternatives", [])
                    
                    st.divider()
                    st.success(f"**Medicine Found:** {price_data.get('medicine_name')}")
                    st.caption(f"**Composition:** `{composition.upper()}`")
                    
                    # Metrics Display
                    st.subheader("📊 Price Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Actual Brand Price", f"₹{actual_price:.2f}")
                    with col2:
                        st.metric("AI Predicted Fair Price", f"₹{fair_price:.2f}")
                    with col3:
                        diff = actual_price - fair_price
                        if diff > 0:
                            st.metric("Price Status", "Overpriced", delta=f"-₹{abs(diff):.2f}", delta_color="inverse")
                        else:
                            st.metric("Price Status", "Fair / Below Average", delta=f"+₹{abs(diff):.2f}", delta_color="normal")
                    
                    st.divider()
                    
                    # Alternatives & Chart
                    st.subheader("📉 Cheaper Alternatives")
                    
                    if isinstance(alternatives, list) and len(alternatives) > 0:
                        df_alts = pd.DataFrame(alternatives)
                        
                        st.dataframe(
                            df_alts.rename(columns={
                                "brand_name": "Brand Name", 
                                "manufacturer": "Manufacturer",
                                "price_inr": "Price (₹)",
                                "pack_size": "Pack Size",
                                "pack_unit": "Pack Unit"
                            }), 
                            use_container_width=True
                        )
                        
                        st.subheader("🏢 Manufacturer Price Comparison")
                        chart_data = df_alts[['manufacturer', 'price_inr']].copy()
                        st.bar_chart(data=chart_data, x="manufacturer", y="price_inr", color="#2e7bcf")
                        
                    else:
                        st.warning("No cheaper alternative brands found for this exact composition.")
                        
                elif price_res.status_code == 404:
                    st.error("❌ Medicine not found in database. Please check spelling or try a sample from the sidebar.")
                else:
                    st.error(f"❌ Server Error: {price_res.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI backend. Make sure your FastAPI server is running!")
    else:
        st.warning("⚠️ Please type a medicine name, scan a photo, or select a sample from the sidebar.")