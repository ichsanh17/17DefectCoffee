import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tempfile
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Coffee Defect Analyzer",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    """
<style>
    /* For class .st-emotion-cache-13k62yr */
    .st-emotion-cache-13k62yr {
        color: #3b2c1e !important;  /* Coffee color for text */
    }
    .st-emotion-cache-ltfnpr{
         color: #3b2c1e !important;  /* Coffee color for text */
    }
    .st-emotion-cache-r90ti5{
        background-color: #6b4f31;
    }
    .block-container {
        padding-top: 50px !important;
        padding-bottom: 50px !important;
        padding-left: 50px !important;
        padding-right: 50px !important;
    }

    /* If the user prefers dark theme, change text color to white */
    @media (prefers-color-scheme: dark) {
        .st-emotion-cache-13k62yr,
        .st-emotion-cache-ltfnpr {
            color: #ffffff !important;  /* White color for dark mode */
        }

        .st-emotion-cache-r90ti5 {
            background-color: #333333;  /* Darker background for dark mode */
        }

    /* For sidebar content with data-testid="stSidebarContent" */
    [data-testid="stSidebarContent"] {
        color: #ffffff !important;  /* White text color */
    }

    # .block-container{
    #     padding-top: 50px;
    #     padding-bottom: 50px;
    #     padding-left: 50px;
    #     padding-right: 50px;
    # }

    /* Custom styles for other elements (if needed) */
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #855930;  /* Dark brown for better contrast */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #bfa382;  /* Slightly lighter brown for tabs */
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #f5f5f5;  /* Light text color for better contrast with tab background */
    }
    .stTabs [aria-selected="true"] {
        background-color: #6b4f31;  /* Darker brown for selected tab */
        color: #f5f5f5;  /* Light text color when selected */
    }
    .stTabs [aria-selected="false"] {
        background-color: #bfa382;  /* Lighter brown for unselected tab */
        color: #5c4033;  /* Darker text for unselected tabs */
    }
    .upload-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-section {
        background-color: #a57c5a;  /* Rich brown for result section */
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .defect-card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.3s;
        color: #3b2c1e;  /* Dark brown text */
    }
    .defect-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e2ded5;  /* Light gray background for info box */
        border-left: 5px solid #6b4f31;  /* Dark brown border for contrast */
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
        color: #3b2c1e;  /* Dark brown text */
    }
    .sidebar .sidebar-content {
        background-color: #f0e6d2;  /* Soft beige for sidebar */
        color: #3b2c1e;  /* Dark brown text */
    }
    .stButton>button {
        background-color: #6b4f31;  /* Darker brown for buttons */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #7d5a4d;  /* Slightly lighter brown for hover state */
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 10px 0;
    }
    .mobile-friendly {
        display: flex;
        flex-direction: column;
    }
    @media (min-width: 768px) {
        .mobile-friendly {
            flex-direction: row;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# Disable TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Function to preprocess image for model
def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess image for EfficientNetB0

    Args:
        img: PIL Image object
        target_size: Target size for resize

    Returns:
        preprocessed_img: Preprocessed image
    """
    # Resize image
    img = img.resize(target_size)

    # Convert to array
    img_array = img_to_array(img)

    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess for EfficientNet
    preprocessed_img = preprocess_input(img_array)

    # Convert to float32 (for TFLite)
    preprocessed_img = preprocessed_img.astype(np.float32)

    return preprocessed_img


# Function to load TFLite model
def load_tflite_model(model_path):
    """
    Load TFLite model

    Args:
        model_path: Path to .tflite model file

    Returns:
        interpreter: TFLite interpreter
    """
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


# Function to predict with TFLite
def predict_with_tflite(interpreter, input_details, output_details, preprocessed_img):
    """
    Predict with TFLite model

    Args:
        interpreter: TFLite interpreter
        input_details: Input details
        output_details: Output details
        preprocessed_img: Preprocessed image

    Returns:
        predictions: Prediction results
    """
    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], preprocessed_img)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    predictions = interpreter.get_tensor(output_details[0]["index"])

    return predictions


# Function to create occlusion heatmap
def make_occlusion_heatmap(img_array, interpreter, input_details, output_details):
    """
    Create heatmap using occlusion sensitivity

    Args:
        img_array: Input image array
        interpreter: TFLite interpreter
        input_details: Input details
        output_details: Output details

    Returns:
        heatmap: Heatmap for visualization
    """
    # Get original predictions
    original_preds = predict_with_tflite(
        interpreter, input_details, output_details, img_array
    )
    pred_class = np.argmax(original_preds[0])
    original_confidence = original_preds[0, pred_class]

    # Create empty heatmap
    heatmap_size = (7, 7)  # Common for EfficientNet
    heatmap = np.zeros(heatmap_size)

    # Patch size for occlusion
    img_size = img_array.shape[1:3]
    patch_size = (img_size[0] // 7, img_size[1] // 7)

    # Apply occlusion across the entire image with 7x7 grid
    for i in range(heatmap_size[0]):
        y_start = i * patch_size[0]
        y_end = min((i + 1) * patch_size[0], img_size[0])

        for j in range(heatmap_size[1]):
            x_start = j * patch_size[1]
            x_end = min((j + 1) * patch_size[1], img_size[1])

            # Create image copy
            masked_img = np.copy(img_array)

            # Apply black patch
            masked_img[0, y_start:y_end, x_start:x_end, :] = 0

            # Predict with partially occluded image
            preds = predict_with_tflite(
                interpreter, input_details, output_details, masked_img
            )
            confidence = preds[0, pred_class]

            # Calculate difference with original prediction
            # Larger difference means more important area
            diff = original_confidence - confidence
            heatmap[i, j] = diff

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat > 0:
        heatmap = heatmap / max_heat

    return heatmap


# Function to overlay heatmap on original image
def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image

    Args:
        img: Original image (PIL Image)
        heatmap: Occlusion heatmap
        alpha: Alpha factor for blending

    Returns:
        original_img: Original image
        heatmap_img: Resized heatmap
        superimposed_img: Image with heatmap overlay
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert heatmap to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed_img = heatmap_colored * alpha + img_array * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)

    return img_array, heatmap_resized, superimposed_img


# Function to create image with matplotlib and convert to base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str


# Define coffee defect information
coffee_defects = {
    "Broken": {
        "description": "Broken beans are coffee beans that have been fractured or broken during processing, typically during hulling or handling.",
        "causes": "Aggressive mechanical processing, improper adjustment of equipment, or low moisture content before hulling.",
        "impact": "May result in inconsistent roasting and flavor development or more rapid staling.",
        "prevention": "Proper adjustment of processing equipment and maintaining optimal moisture content.",
        "example_image": "example/Broken_01.jpg",
    },
    "Cut": {
        "description": "Cut beans are beans that have been physically damaged or torn, which impacts their visual appearance and quality.",
        "causes": "Improper handling or mechanical damage during processing.",
        "impact": "Inconsistent roasting, and it can negatively affect the flavor profile of the coffee.",
        "prevention": "Proper handling and calibration of processing equipment.",
        "example_image": "example/Cut_01.jpg",
    },
    "Dry Cherry": {
        "description": "Dry cherry beans are coffee beans that have not been adequately processed and dried, resulting in beans that have high moisture content.",
        "causes": "Inadequate drying procedures or exposure to excess moisture during processing.",
        "impact": "May result in mold growth or uneven roasting.",
        "prevention": "Ensure beans are properly dried to the correct moisture content.",
        "example_image": "example/Dry Cherry_01.jpg",
    },
    "Fade": {
        "description": "Faded beans appear whitish or pale in color instead of the normal greenish tint of unroasted coffee.",
        "causes": "Age, exposure to moisture after drying, or prolonged storage in unsuitable conditions.",
        "impact": "Flat, papery, or aged flavors in the cup due to oxidation and loss of flavor compounds.",
        "prevention": "Proper storage in cool, dry conditions with appropriate packaging.",
        "example_image": "example/Fade_01.jpg",
    },
    "Floater": {
        "description": "Floater beans are beans that float when placed in water, indicating they have a lower density and are often underdeveloped or damaged.",
        "causes": "Immature cherries or poor processing conditions.",
        "impact": "Can cause uneven roasting, affecting overall flavor quality.",
        "prevention": "Selective harvesting of ripe cherries and proper processing techniques.",
        "example_image": "example/Floater_01.jpg",
    },
    "Full Black": {
        "description": "Full black beans are coffee beans that are black in color, typically resulting from over-fermentation, disease, or improper drying.",
        "causes": "Over-fermentation, fungal infection, or improper drying.",
        "impact": "Creates bitter, unpleasant tastes in coffee with burnt or fermented flavors.",
        "prevention": "Ensure proper fermentation time, maintain cleanliness during processing, and adequate drying.",
        "example_image": "example/Full Black_01.jpg",
    },
    "Full Sour": {
        "description": "Full sour beans are those that have a sour or fermented odor, often resulting from improper fermentation processes.",
        "causes": "Improper fermentation or over-fermentation.",
        "impact": "Produces sour, unpleasant flavors in the cup.",
        "prevention": "Proper fermentation methods and monitoring of fermentation times.",
        "example_image": "example/Full Sour_01.jpg",
    },
    "Fungus Damage": {
        "description": "Beans with visible damage from fungal growth, often appearing as discolored patches or spots.",
        "causes": "Fungal infection during cultivation, processing, or storage.",
        "impact": "Can produce moldy, fermented, or phenolic flavors.",
        "prevention": "Fungicide application when necessary, proper drying, and storage.",
        "example_image": "example/Fungus Damage_01.jpg",
    },
    "Husk": {
        "description": "Fragments of the outer skin of the coffee cherry that have not been removed during processing.",
        "causes": "Inadequate pulping or dry processing techniques.",
        "impact": "Can cause dirty or earthy flavors in the cup.",
        "prevention": "Proper pulping equipment maintenance and calibration.",
        "example_image": "example/Husk_01.jpg",
    },
    "Immature": {
        "description": "Immature beans come from coffee cherries harvested before they reach full ripeness, resulting in underdeveloped beans.",
        "causes": "Harvesting unripe cherries or severe drought affecting cherry development.",
        "impact": "Harsh, astringent flavors with heightened acidity and lack of sweetness.",
        "prevention": "Selective harvesting of only ripe cherries and proper sorting.",
        "example_image": "example/Immature_01.jpg",
    },
    "Parchment": {
        "description": "Parchment refers to beans where the silver skin or parchment layer has not been completely removed during processing.",
        "causes": "Insufficient hulling or mechanical processing issues.",
        "impact": "Can create an inconsistent roast and potentially chaffy flavors in the cup.",
        "prevention": "Proper adjustment and maintenance of hulling equipment.",
        "example_image": "example/Parchment_01.jpg",
    },
    "Partial Black": {
        "description": "Partial black beans are beans that have some blackened areas due to over-fermentation or disease.",
        "causes": "Over-fermentation, fungal infection, or improper drying.",
        "impact": "May result in burnt, bitter flavors in the coffee.",
        "prevention": "Ensure proper drying and processing conditions.",
        "example_image": "example/Partial Black_01.jpg",
    },
    "Partial Sour": {
        "description": "Partial sour beans have a slightly sour or fermented odor, often resulting from inadequate fermentation.",
        "causes": "Improper fermentation or incomplete processing.",
        "impact": "Can lead to off-flavors such as sourness in the cup.",
        "prevention": "Ensure proper fermentation and processing methods.",
        "example_image": "example/Partial Sour_01.jpg",
    },
    "Severe Insect Damage": {
        "description": "Beans with severe insect damage, often characterized by large visible holes or significant degradation.",
        "causes": "Infestation by coffee berry borers or other pests.",
        "impact": "Insect-damaged beans contribute musty or fermented flavors to the cup.",
        "prevention": "Integrated pest management practices, prompt harvesting, and proper storage.",
        "example_image": "example/Severe Insect Damange_01.jpg",
    },
    "Shell": {
        "description": "Shell-shaped beans resulting from the development of only one side of the coffee seed.",
        "causes": "Genetic factors or abnormal development of the coffee cherry.",
        "impact": "May cause uneven roasting, affecting cup consistency.",
        "prevention": "Cannot be fully prevented as it's partly genetic, but proper growing conditions help.",
        "example_image": "example/Shell_01.jpg",
    },
    "Slight Insect Damage": {
        "description": "Beans with minor insect damage, usually small holes or slight degradation.",
        "causes": "Light infestation by pests like coffee berry borers.",
        "impact": "May lead to slight off-flavors in the coffee.",
        "prevention": "Pest control and proper processing.",
        "example_image": "example/Slight Insect Damage_01.jpg",
    },
    "Withered": {
        "description": "Withered beans appear shriveled and smaller than normal due to inadequate development.",
        "causes": "Drought conditions, plant stress, or nutrient deficiencies during cherry development.",
        "impact": "Can contribute astringent or woody flavors to the cup.",
        "prevention": "Proper irrigation, fertilization, and shade management in coffee cultivation.",
        "example_image": "example/Withered_01.jpg",
    },
}

# App title and introduction
st.title("☕ Coffee Defect Analyzer")
st.markdown(
    """
<div class="info-box">
This application helps identify defects in coffee beans using artificial intelligence. 
Upload an image of coffee beans to determine the type of defect present. The app also visualizes 
how the AI makes its decision using Grad-CAM technology.
</div>
""",
    unsafe_allow_html=True,
)

# Create tabs for different sections of the app
tab1, tab2, tab3 = st.tabs(
    ["🔍 Analyze Coffee Defects", "📚 Defect Library", "ℹ️ About Grad-CAM"]
)

with tab1:
    st.header("Analyze Your Coffee Beans")

    # File uploader
    # st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a coffee bean image...", type=["jpg", "jpeg", "png"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Check for model file
    model_path = "efficientNet_Model_Quantized_Float16 (1).tflite"
    model_exists = os.path.exists(model_path)

    if not model_exists:
        st.warning(
            "⚠️ Model file not found. Please make sure the model is placed in the correct location."
        )
        st.markdown(
            """
        <div class="info-box">
        Expected path: <code>efficientNet_Model_Quantized_Float16 (1).tflite</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Class names (sorted alphabetically to match model's expectations)
    class_names = sorted(
        [
            "Broken",
            "Cut",
            "Dry Cherry",
            "Fade",
            "Floater",
            "Full Black",
            "Full Sour",
            "Fungus Damage",
            "Husk",
            "Immature",
            "Parchment",
            "Partial Black",
            "Partial Sour",
            "Severe Insect Damage",
            "Shell",
            "Slight Insect Damage",
            "Withered",
        ]
    )

    if uploaded_file is not None and model_exists:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        # Display progress
        with st.spinner("Analyzing image..."):
            # Load and preprocess image
            image = Image.open(uploaded_file).convert("RGB")

            # Create a 3-column layout
            col1, col2 = st.columns(2)

            with col1:
                # Display the original image
                st.subheader("Uploaded Image")
                st.image(image, caption="Original Image", use_container_width=True)

            with col2:
                # Load model
                interpreter, input_details, output_details = load_tflite_model(
                    model_path
                )

                # Preprocess image for model
                preprocessed_img = preprocess_image(image)

                # Get prediction
                predictions = predict_with_tflite(
                    interpreter, input_details, output_details, preprocessed_img
                )
                pred_idx = np.argmax(predictions[0])
                pred_class = class_names[pred_idx]
                confidence = predictions[0][pred_idx] * 100

                # Display prediction
                st.subheader("Analysis Result")
                st.write(f"**Detected Defect:** {pred_class.replace('_', ' ')}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                # Get defect information
                if pred_class in coffee_defects:
                    defect_info = coffee_defects[pred_class]

                    # Create expandable section for defect details
                    with st.expander("See defect details", expanded=True):
                        st.markdown(f"**Description:** {defect_info['description']}")
                        st.markdown(f"**Causes:** {defect_info['causes']}")
                        st.markdown(
                            f"**Impact on Coffee Quality:** {defect_info['impact']}"
                        )
                        st.markdown(f"**Prevention:** {defect_info['prevention']}")

            # # Display the original image
            # st.subheader("Uploaded Image")
            # st.image(image, caption="Original Image", use_container_width=True)

            # # Load model
            # interpreter, input_details, output_details = load_tflite_model(model_path)

            # # Preprocess image for model
            # preprocessed_img = preprocess_image(image)

            # # Get prediction
            # predictions = predict_with_tflite(
            #     interpreter, input_details, output_details, preprocessed_img
            # )
            # pred_idx = np.argmax(predictions[0])
            # pred_class = class_names[pred_idx]
            # confidence = predictions[0][pred_idx] * 100

            # # Display prediction
            # st.subheader("Analysis Result")
            # st.write(f"**Detected Defect:** {pred_class.replace('_', ' ')}")
            # st.write(f"**Confidence:** {confidence:.2f}%")

            # # Get defect information
            # if pred_class in coffee_defects:
            #     defect_info = coffee_defects[pred_class]

            #     # Create expandable section for defect details
            #     with st.expander("See defect details", expanded=True):
            #         st.markdown(f"**Description:** {defect_info['description']}")
            #         st.markdown(f"**Causes:** {defect_info['causes']}")
            #         st.markdown(
            #             f"**Impact on Coffee Quality:** {defect_info['impact']}"
            #         )
            #         st.markdown(f"**Prevention:** {defect_info['prevention']}")

            # Generate Grad-CAM visualization
            st.subheader("AI Focus Areas (Grad-CAM)")

            # Create occlusion heatmap
            heatmap = make_occlusion_heatmap(
                preprocessed_img, interpreter, input_details, output_details
            )

            # Create visualizations
            original_img, heatmap_img, superimposed_img = save_and_display_gradcam(
                image, heatmap
            )

            # Create a three-column layout for visualizations
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(original_img, caption="Original", use_container_width=True)

            with col2:
                # Display the heatmap with colormap
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(heatmap, cmap="jet")
                ax.set_title("Heatmap")
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)

            with col3:
                st.image(superimposed_img, caption="Overlay", use_container_width=True)

            # Explanation of the visualization
            st.markdown(
                """
            <div class="info-box">
            <strong>How to interpret these visualizations:</strong><br>
            - <strong>Original:</strong> Your uploaded image<br>
            - <strong>Heatmap:</strong> Areas the AI focuses on to make its decision (red = most important)<br>
            - <strong>Overlay:</strong> Heatmap superimposed on the original image
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("Coffee Defect Library")
    st.write(
        "Explore the 17 types of coffee defects with descriptions and example images."
    )

    # Search box for defects
    search_term = st.text_input("Search for a defect type", "")

    # Filter defects based on search term
    filtered_defects = {}
    if search_term:
        for defect_name, defect_info in coffee_defects.items():
            if (
                search_term.lower() in defect_name.lower()
                or search_term.lower() in defect_info["description"].lower()
            ):
                filtered_defects[defect_name] = defect_info
    else:
        filtered_defects = coffee_defects

    # Display defects in a grid
    col1, col2 = st.columns(2)

    for i, (defect_name, defect_info) in enumerate(filtered_defects.items()):
        # Alternate between columns
        with col1 if i % 2 == 0 else col2:
            st.markdown(f'<div class="defect-card">', unsafe_allow_html=True)
            st.subheader(defect_name.replace("_", " "))

            # Use a placeholder image if the example_image is not available
            try:
                st.image(
                    defect_info["example_image"],
                    caption=f"{defect_name.replace('_', ' ')} Example",
                    use_container_width=True,
                )
            except:
                st.info("Example image not available")

            st.markdown(f"**Description:** {defect_info['description']}")

            with st.expander("See more details"):
                st.markdown(f"**Causes:** {defect_info['causes']}")
                st.markdown(f"**Impact on Coffee Quality:** {defect_info['impact']}")
                st.markdown(f"**Prevention:** {defect_info['prevention']}")

            st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.header("Understanding Grad-CAM Technology")

    st.markdown(
        """
    <div class="info-box">
    <h3>What is Grad-CAM?</h3>
    Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique for making Convolutional Neural Networks (CNNs) more transparent by visualizing which regions of an image are important for predictions.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### How Grad-CAM Works
    
    In this application, we use a variation called Occlusion Sensitivity which:
    
    1. Takes a trained model and an input image
    2. Systematically covers (occludes) different portions of the image
    3. Observes how the prediction changes with each occlusion
    4. Creates a heatmap where red areas show regions that strongly influence the model's decision
    
    ### Why This Matters for Coffee Defect Analysis
    
    Grad-CAM helps:
    
    - **Verify Focus:** Ensures the AI is looking at the actual defect and not background elements
    - **Increase Trust:** Shows how the model makes decisions, making the process transparent
    - **Improve Analysis:** Helps experts understand which visual features are important for classification
    - **Detect Biases:** Reveals if the model is using irrelevant visual cues for predictions
    
    ### Interpreting the Heatmap
    
    - **Red/Yellow Areas:** Highly influential for the model's decision
    - **Blue/Green Areas:** Less important for classification
    - **No Color (Blue):** Minimal influence on the prediction
    """
    )

    # Add example of grad-cam
    st.image(
        "https://miro.medium.com/max/1400/1*6gdpKnDRRHDN9YonVCrjjw.jpeg",
        caption="Example of Grad-CAM visualization (Source: Medium.com)",
        use_container_width=True,
    )

# Add footer
st.markdown(
    """
---
<p style="text-align: center;">
    Coffee Defect Analyzer © 2025 | Created with Streamlit and TensorFlow
</p>
""",
    unsafe_allow_html=True,
)

# Add sidebar with additional information
with st.sidebar:
    st.header("Application Info")
    st.markdown(
        """
    This application uses a trained EfficientNet model to identify 17 different types of coffee bean defects.
    
    **Features:**
    - Defect identification
    - Confidence score
    - Grad-CAM visualization
    - Comprehensive defect library
    
    **Model Information:**
    - Architecture: EfficientNet
    - Input Size: 224x224
    - Preprocessing: EfficientNet standard
    """
    )

    # Add version info
    st.sidebar.markdown("---")
    st.sidebar.caption("Version 1.0")

    # Add usage instructions
    st.sidebar.markdown("## How to Use")
    st.sidebar.markdown(
        """
    1. Go to the "Analyze Coffee Defects" tab
    2. Upload a clear image of coffee beans
    3. View the detected defect and explanation
    4. Explore the heatmap to understand AI decision making
    5. Browse the defect library for more information
    """
    )
