**CSE 366 — Artificial Intelligence**
**Group Assignment: Streamlit App with Model Selection & XAI Visualizations**

Dataset: teaLeafBD (https://data.mendeley.com/datasets/744vznw5k2/4)

Features implemented in the streamlit app:

**Model Selection**: Choose among custom CNN, ResNet-152, DenseNet-201 and EfficientNet-B3.
**Image Input**: Upload your own image or select from provided samples from each disease class.
**Prediction**: Get the top 3 predicted classes with their confidence scores for 7 different tea leaf conditions.
**XAI Visualizations**: Generate and compare five different explanations (Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, LIME)
**Export**: Download all generated explanation images as a .zip file.

**Setup steps(Assuming for windows):**

1.  Create a Virtual Environment:
    **bash**
    python -m venv venv
    venv\Scripts\activate

2.  Install Dependencies:
    **bash**
    pip install -r requirements.txt

3.  Download the dataset from [https://data.mendeley.com/datasets/744vznw5k2/4] and place it in the same folder as the .py files.

4.  Download and Place Model Weights:
    Place the trained model weights into the `/weights` directory. These are the files that would store the weights:
    1. custom_cnn.pth
    2. resnet152.pth
    3. densenet201.pth
    4. efficientnet_b3.pth

**Run the app and feature exploration**

1.  Run the Streamlit App:
    **bash**
    streamlit run app.py

2.  Open the URL provided by Streamlit in your browser.

3. Use the sidebar to select a model and an input image.

4. Click the "Generate All Explanations" button to view the XAI visualizations.

5. Download all generated explanation images as a .zip file by clicking "Download all generated visualizations as a single ZIP file."

**MODEL LIST**

1. Custom CNN

2. Resnet152

3. Densenet201

4. Efficientnet_b3

**Team members - roles**

1. Mahmuda Kahnam Niha(2022-2-60-030) - Implemented the XAI visualization module using grad-cam and lime, wrote the README.md, developed the Streamlit UI layout.
2. Zannatul Naeem Shoron(2022-2-60-112) - Trained the Custom CNN, ResNet-152, DenseNet-201 and EfficientNet-B3, implemented the model loading and prediction logic, created the ZIP export feature.


**Citation**
1. **Dataset**: TLD-24: a comprehensive dataset of tea leaf diseases and pests by Chakma, R., et al. (2024) on Mendeley Data, V1. (https://data.mendeley.com/datasets/744vznw5k2/4)
2. **External Assets**: Pre-trained weights for ResNet, DenseNet, and EfficientNet were sourced from `torchvision.models`, taken Streamlit App inspiration from Gemini.