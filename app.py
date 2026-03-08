import os
import streamlit as st
from PIL import Image
from torchvision import transforms
import timm
import torch

# Load model
## Check gpu availability with pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
## Load your model from the Hub
model_reloaded = timm.create_model(
    "put your own checkpoint here ...",
    pretrained=True
)
model = model_reloaded.eval()
model = model.to(device)

# Define labels
class_name = ["apple pie", "bibimbap", "cannoli", "edamame", "falafel", "french toast", "ice cream", "ramen", "sushi", "tiramisu"]

# Define the sample images
sample_images = {
    "Apple Pie": "./notebooks/test_images/Apple-Pie-Recipe-Recipe-Card.jpg",
    "Ice Cream": "./notebooks/test_images/no-churn-ice-cream-four-ways-15139-2.jpg",
    "Ramen": "./notebooks/test_images/Spicy-Shoyu-Ramen-8055-I.jpg",
}

def transform_image(image):
    # Define transformations to apply to the images
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),           # Convert images to PyTorch tensors
    ])
    # Apply transformations and move tensor to device
    return transform(image).unsqueeze(0).to(device)


def predict(image):
    try:
        image = transform_image(image)
        with torch.no_grad():
            # Perform prediction
            outputs = model(image)
            # Apply softmax to convert scores to probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top k probabilities and labels
            k = 5
            topk_probs, topk_indices = torch.topk(probs, k)
            topk_probs = topk_probs.squeeze().cpu().numpy()
            topk_indices = topk_indices.squeeze().cpu().numpy()

            # Combine probabilities and labels into a list of tuples
            topk_results = [(prob, label) for prob, label in zip(topk_probs, topk_indices)]
            return topk_results
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []

# Define the Streamlit app
def app():
    st.title("Foods Classification")

    # Add a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # # Add a selectbox to choose from sample images
    sample = st.selectbox("Or choose from sample images:", list(sample_images.keys()))

    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        predictions = predict(image)

    # If a sample image is chosen, make a prediction on it
    elif sample:
        image = Image.open(sample_images[sample])
        st.image(image, caption=sample.capitalize() + " Image.", use_column_width=True)
        predictions = predict(image)

    # Show the top 3 predictions with their probabilities
    if predictions:
        st.write("Top 3 predictions:")
        for i, (prob, label) in enumerate(predictions):
            st.write(f"{i+1}. {class_name[label]} ({prob*100:.2f}%)")

            # Show progress bar with probabilities
            st.markdown(
                """
                <style>
                .stProgress .st-b8 {
                    background-color: orange;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(prob))  # Convert prob to float before passing it to st.progress()
    else:
        st.write("No predictions.")

# Run the app
if __name__ == "__main__":
    app()
