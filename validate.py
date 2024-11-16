import mlflow.pytorch
import torch
import shap
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Initialize the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model from MLflow
run_id = 'b691110f9e8142b38993b3c4bb12a38e'  # Your existing run ID
model_uri = f'runs:/{run_id}/model'  
loaded_model = mlflow.pytorch.load_model(model_uri)
loaded_model.eval()

# Create a wrapper function for the model that handles numpy arrays
def model_wrapper(input_data):
    input_tensor = torch.FloatTensor(input_data).to(device)
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.reshape(-1, 1, 28, 28)
    with torch.no_grad():
        output = loaded_model(input_tensor)
    return output.cpu().numpy()

# Prepare the input data
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
input_example, _ = test_dataset[0]

# Create background data for SHAP
n_background = 5
background_data = []
for i in range(n_background):
    img, _ = test_dataset[i]
    background_data.append(img.numpy().flatten())
background_data = np.array(background_data)

# Initialize SHAP explainer and calculate values
explainer = shap.KernelExplainer(model_wrapper, background_data)
input_data = input_example.numpy().flatten().reshape(1, -1)
shap_values = explainer.shap_values(input_data)

# Print shape information
print(f"Input shape: {input_data.shape}")
print(f"Number of classes: {len(shap_values)}")
print(f"SHAP values shape for each class: {[sv.shape for sv in shap_values]}")

# Create visualization
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(131)
plt.imshow(input_example.squeeze().numpy(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Calculate and reshape SHAP values
shap_values_array = np.array(shap_values)
total_shap = np.abs(shap_values_array).sum(axis=0)
print(f"Total SHAP values shape before reshape: {total_shap.shape}")
total_shap = total_shap.reshape(28, 28, 10).mean(axis=2)

# SHAP magnitude
plt.subplot(132)
plt.imshow(total_shap, cmap='hot')
plt.title('SHAP Values Magnitude')
plt.colorbar()
plt.axis('off')

# SHAP overlay
plt.subplot(133)
plt.imshow(input_example.squeeze().numpy(), cmap='gray')
plt.imshow(total_shap, cmap='hot', alpha=0.5)
plt.title('SHAP Overlay')
plt.axis('off')

# Save plots
plt.tight_layout()
combined_image_path = "input_and_shap_explanation.png"
plt.savefig(combined_image_path)

# Log to the existing MLflow run
client = mlflow.tracking.MlflowClient()
with mlflow.start_run(run_id=run_id):
    # Log SHAP visualization
    mlflow.log_artifact(combined_image_path, "shap_explanations")
    
    # Log raw SHAP values
    np.save('shap_values.npy', shap_values)
    mlflow.log_artifact('shap_values.npy', "shap_explanations")
    
    # Log metadata
    mlflow.log_params({
        'shap_n_background_samples': n_background,
        'shap_input_shape': str(input_example.shape),
        'shap_n_output_classes': len(shap_values)
    })
    
    # Log some summary statistics about the SHAP values
    mlflow.log_metrics({
        'shap_mean_magnitude': float(np.abs(total_shap).mean()),
        'shap_max_magnitude': float(np.abs(total_shap).max())
    })
    
    # Log feature importance summary
    feature_importance = np.abs(total_shap).mean()
    mlflow.log_metric('average_feature_importance', float(feature_importance))

print("SHAP analysis logged to existing MLflow run:", run_id)
