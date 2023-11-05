# Importing the libraries needed
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
import joblib
from efficientnet_pytorch import EfficientNetForImageClassification

# Defining the transformation layer
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Loading the class-to-indices map
CLASS_INDICES = joblib.load('class_indices.pkl')

# Reversing the class-to-indices map
INDICES_CLASS = {v: k for k, v in CLASS_INDICES.items()}

# Loading the model
ENB2 = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b2",
                                                           num_labels=525,
                                                           id2label=INDICES_CLASS,
                                                           label2id=CLASS_INDICES,
                                                           ignore_mismatched_sizes=True
                                                         )

# Changing the output layer
ENB2.classifier = torch.nn.Linear(in_features=1408,
                                  out_features=525, bias=True)

# Loading the model weights
ENB2.load_state_dict(torch.load("models/enb2_525_birds.pt"))

# Creating an inference function
def model_inference(model, image, indices_class,
                    transform, top_k=5):

    # Setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sending the model to device
    model = model.to(device)

    # Setting the model to evaluation mode
    model.eval()

    # Transforming the image
    image = transform(image).unsqueeze(0)

    # Moving the image to the device
    image = image.to(device)

    # Getting the predictions
    with torch.no_grad():
        predictions = model(image)['logits']
        predictions = F.softmax(predictions, dim=-1)

    # Getting the top k predictions
    top_k_predictions = torch.topk(predictions, k=top_k, dim=1)

    # Getting the top k probabilities and indices
    top_k_probabilities = top_k_predictions.values.squeeze().tolist()
    top_k_indices = top_k_predictions.indices.squeeze().tolist()

    # Getting the top k classes
    top_k_classes = [indices_class[i] for i in top_k_indices] \
                    if type(top_k_indices) != int else indices_class[top_k_indices]

    # Returning the top k probabilities and classes
    return top_k_probabilities, top_k_classes


if __name__ == "__main__":
    # Creating an argument parser
    parser = argparse.ArgumentParser()

    # Adding the arguments
    parser.add_argument("-i", "--image", required=True, help="Path to the image")
    parser.add_argument("-k", "--top_k", required=False, default=5, help="Top k probabilities and classes")

    # Parsing the arguments
    args = parser.parse_args()

    # Loading the image
    image = Image.open(args.image)

    # Getting the top k probabilities and classes
    top_k_probabilities, top_k_classes = model_inference(ENB2, image, INDICES_CLASS, 
                                                         transform=IMG_TRANSFORMS, top_k=args.top_k)