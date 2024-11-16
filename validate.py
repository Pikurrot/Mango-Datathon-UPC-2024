import mlflow.pytorch
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# Replace <run_id> with the actual run ID
model_uri = 'runs:/15df3c352fba42518976ef2b9254dfa0/model'  # Example: 'runs:/2b5860efc42e485295b64f1091c5eae7/model'
loaded_model = mlflow.pytorch.load_model(model_uri)

# Set the model to evaluation mode
loaded_model.eval()

# Create a dummy input (for FashionMNIST, the shape should be (1, 28, 28))
# Replace this with your actual input
input_example = torch.randn(1, 28, 28).to(device)  # Example: random tensor, replace as needed

# Perform inference
with torch.no_grad():
    output = loaded_model(input_example)

# Print the model output (logits)
print("Model output:", output)