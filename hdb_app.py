import streamlit as st
import pandas as pd

from scripts.inference import predict_price
from scripts.model_registry import retrieve
from scripts.config import appconfig



def get_region_data():
    return {
        'CENTRAL': ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TOA PAYOH'],
        'NORTH': ['SEMBAWANG', 'WOODLANDS', 'YISHUN'],
        'NORTHEAST': ['ANG MO KIO', 'HOUGANG', 'PUNGGOL', 'SENGKANG', 'SERANGOON'],
        'EAST': ['BEDOK', 'PASIR RIS', 'TAMPINES'],
        'WEST': ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI', 'JURONG EAST', 'JURONG WEST', 'LIM CHU KANG']
    }

def get_flat_type_options():
    return {
        "1-Room": (28, 31),
        "2-Room": (34, 67),
        "3-Room": (46, 307),
        "4-Room": (109, 149),
        "5-Room": (99, 210),
        "Executive": (124, 243),
        "Multi-Generation": (132, 179)
    }

def create_input_dataframe(region, town, flat_type, storey_level, floor_area):
    return pd.DataFrame(
        [[region, town, flat_type, storey_level, floor_area]],
        columns=["region", "town", "flat_type_standardized", "median_storey", "floor_area_sqm"]
    )


def get_image_urls():
    df = pd.read_csv(photo_path)
    image_urls = df["url"].tolist()
    return image_urls

# ----------------------------------------
# Load model (cached once per session)
# ----------------------------------------

@st.cache_resource
def get_ml_model_and_features():
    return retrieve("hdb_price_predictor")

# ----------------------------------------
# App layout
# ----------------------------------------


st.header("HDB Resale Flat Price Predictor")

region_dict = get_region_data()
region = st.sidebar.selectbox("Region", list(region_dict.keys()))
town = st.sidebar.selectbox("Town", region_dict.get(region, []))

flat_type_options = get_flat_type_options()
flat_type = st.sidebar.selectbox("Flat Type", list(flat_type_options.keys()))

min_area, max_area = flat_type_options.get(flat_type, (28, 243))
floor_area = st.sidebar.slider("Floor Area (sq meters)", min_area, max_area, min_area)
storey_level = st.sidebar.slider("Storey Level", 2, 50, 2)

col1, col2 = st.sidebar.columns(2)
if col1.button("Predict"):
    st.session_state["predict_clicked"] = True
    st.session_state["inputs"] = (region, town, flat_type, storey_level, floor_area)

if col2.button("Reset"):
    st.session_state.clear()
    st.rerun()

# ----------------------------------------
# Prediction logic
# ----------------------------------------

if st.session_state.get("predict_clicked"):
    model, features = get_ml_model_and_features()

    if model is not None and features is not None:
        region, town, flat_type, storey_level, floor_area = st.session_state["inputs"]
        input_data = create_input_dataframe(region, town, flat_type, storey_level, floor_area)
        prediction = predict_price(model, features, input_data)

        if prediction is not None:
            st.success(f"### Predicted Price: SGD {prediction:,.0f}")
        else:
            st.error("Prediction failed. Please check your inputs or model.")
    else:
        st.error("Model could not be loaded. Make sure it has been trained and saved.")
        
        
# Fetch and display image slideshow using Owl Carousel
photo_path = appconfig['Paths']['photo_path']
try:
    image_urls = get_image_urls()
    # Generate HTML for Owl Carousel
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.3.4/assets/owl.carousel.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.3.4/assets/owl.theme.default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.3.4/owl.carousel.min.js"></script>
        <style>
            .owl-carousel .item {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .owl-carousel img {
                display: block;
                margin: auto;
                max-width: 500px; /* Adjust width */
                height: 250px; /* Maintain aspect ratio */
            }
        </style>
    </head>
    <body>
        <div class="owl-carousel owl-theme">
    """
    for url in image_urls:
        html_content += f'<div class="item"><img src="{url}" alt="Image"></div>'
    html_content += """
        </div>
        <script>
            $(document).ready(function(){
                $(".owl-carousel").owlCarousel({
                    loop: true,
                    margin: 10,
                    autoplay: true,
                    autoplayTimeout: 3000,
                    autoplayHoverPause: true,
                    items: 3  // Number of images shown at once
                });
            });
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_content, height=600)
except Exception as e:
    st.error(f"Error fetching images: {e}")