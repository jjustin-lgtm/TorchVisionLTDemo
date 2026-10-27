# Lightning Talk Demo Program 1: Car Image/Footage Classifier AKA Totally Not Flock




# Program that classifies a car based off a provided image, computer vision (not real time though...)
# Want to make it so that you can provide an image of a road/highway with lots of cars
# the program will then process the image, and draw boundry boxes around each car its identified
# then on the top of those boundary boxes, it will label the each cars identification.

# Car types: Sedan, Truck, SUV, Coupe, Hatchback, Motorcycle

# import everything needed
import argparse

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import to_tensor

# Data from 
# https://data.mendeley.com/datasets/htsngg9tpc/2
# Boonsirisumpun, Narong; surinta, olarik (2021), “Vehicle Type Image Dataset (Version 2): VTID2”, Mendeley Data, V2, doi: 10.17632/htsngg9tpc.2


#CONSOLE COMMANDS NEEDED TO ENTER
# %run TorchVisionLTDemo.py train dataset --epochs 10
# %run TorchVisionLTDemo.py detect traffic.jpg --output result.jpg


# To train the ResNet-18 model
# type "python TorchVisionLTDemo.py train --data dataset --epochs 10"
# into the console. The "vehicle_model.pth" file is the trained ResNet-18 model.

# An epoch is one complete pass through of the training dataset during training.

# To run the program on a given file, type "python vehicle_ai.py detect --image traffic.jpg --output result.png"
# where "traffic.jpg" is the file you want to test.

# For demo, need to enter into console
# "pip install torch torchvision pillow"
# Torch - PyTorch
# TorchVision - TorchVision
# Pillow - Open-Source library for image processing.

# Need to see if there's a prexisting dataset for car detection (THERE IS)

# ImageFolder sorts folder names alphabetically, so keep the same class order.
CLASSES = ["hatchback","other","pickup", "sedan", "suv"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREPROCESS = models.ResNet18_Weights.DEFAULT.transforms() # Creates a standard preprocessing pipeline
# to help provide consistent images when training the model.



# Two step object identifier, first use a pre-trained model to identify
# all of the cars in the image. 

# TorchVision provides a pretrained Mask R-CNN model trained off the
# COCO dataset. this will be our general car identifier.
# (maskrcnn_resnet50_fpn_v2 is the moedl name)
# Create the ResNet-18 model.
def make_classifier(pretrained=False):
    """Create ResNet-18 with four vehicle-category outputs."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)


# Understand what Transforms in in TorchVision.

# Figure out which model is best for car object detection

# (OPS) Custom operators specific to models.

# Look at object detection pre-trained models

# train function actually trains the ResNet-18 model with the
# provided images from the folder parameter.
def train(folder, epochs):
    """Fine-tune ResNet-18 and save its best-performing weights."""
    
    # Image augmentation, to edit each image so that they're consistant
    # and can be properly used to train the model.
    augment = transforms.Compose([
        # By adding these two random transforms, we prevent the model from being "overtuned" to 
        # very specific cases of vehicle orientation and direction.
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        # Next convert the image to the single layer PyToch tensor.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load the training and evaluation data into the program
    train_data = datasets.ImageFolder(f"{folder}/train", transform=augment)
    val_data = datasets.ImageFolder(f"{folder}/val", transform=PREPROCESS)
    
    if train_data.classes != CLASSES or val_data.classes != CLASSES:
        raise ValueError(f"Both train/ and val/ must contain: {CLASSES}")

    print(f"Training on {len(train_data)} images; validating on {len(val_data)} images", flush=True)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    # Initalize the neural network
    model = make_classifier(pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss() # Loss function is used to measure the "reward/punishment" the nn gets if it incorrectly guesses a test.
    
    best_accuracy = -1

    # This is the actual training loop repeated for the number of epochs set.
    # Works in 4 steps:
    # Foward Pass, Calculate Loss, Backpropagation, and Optimizer step
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()  # Update weights based on the prediction error.
            optimizer.step()
            total_loss += loss.item() * len(labels)

    # Then finally, evalulate the trained model using the validation dataset
        model.eval()
        correct = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images.to(DEVICE)).argmax(dim=1)
                correct += (predictions == labels.to(DEVICE)).sum().item()

        accuracy = correct / len(val_data)
        
        # Print results of the training, and then save the trained model.
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"loss: {total_loss / len(train_data):.3f} | "
              f"validation accuracy: {accuracy:.1%}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "vehicle_model.pth")

    print("Best model saved as vehicle_model.pth")

# MaskRCNN

# Define a new class that inherits from neural network module

    # Flatten layer - takes multi-dimensional input and turn it into one layer.
    # use neural network sequential.


# Train the model using images organized into folders for each image type

# Then use the ImageFolder class to load these folders and assign labels

# Detect function is now the actual work of the model,
# used to indetify the type of cars in the model.
def detect(image_path, output, threshold):
    """Find vehicles with Mask R-CNN and label each one using ResNet-18."""
    model = make_classifier()
    model.load_state_dict(torch.load("vehicle_model.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    print("Loading Mask R-CNN (may download weights on first use)...", flush=True)
    detector = models.detection.maskrcnn_resnet50_fpn_v2(
        weights=models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ).to(DEVICE).eval()

    image = Image.open(image_path).convert("RGB")
    with torch.inference_mode():
        found = detector([to_tensor(image).to(DEVICE)])[0]

    result = image.copy()
    draw = ImageDraw.Draw(result)
    count = 0

    for box, label, score in zip(found["boxes"], found["labels"], found["scores"]):
        # COCO class IDs: car=3, bus=6, truck=8.
        if label.item() not in (3, 6, 8) or score.item() < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        with torch.inference_mode():
            probabilities = model(PREPROCESS(crop).unsqueeze(0).to(DEVICE)).softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        count += 1
        text = f"ID {count:03d} | {CLASSES[index.item()]} {confidence.item():.0%}"

        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        bounds = draw.textbbox((0, 0), text)
        label_width = bounds[2] - bounds[0] + 8
        label_height = bounds[3] - bounds[1] + 6
        left = min(x1, max(0, image.width - label_width))
        top = max(0, y1 - label_height)
        draw.rectangle((left, top, left + label_width, top + label_height), fill="lime")
        draw.text((left + 4, top + 3 - bounds[1]), text, fill="black")
        count += 1

    result.save(output)
    print(f"Saved {output} with {count} detected vehicles")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run a vehicle detector")
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train the ResNet-18 classifier")
    training.add_argument("dataset", help="Folder containing train/ and val/")
    training.add_argument("--epochs", type=int, default=10)

    detection = commands.add_parser("detect", help="Detect and classify vehicles")
    detection.add_argument("image", help="Input photograph")
    detection.add_argument("--output", default="result.jpg")
    detection.add_argument("--threshold", type=float, default=0.6)

    args = parser.parse_args()
    print(f"Using device: {DEVICE}", flush=True)
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        train(args.dataset, args.epochs)
    else:
        detect(args.image, args.output, args.threshold)


