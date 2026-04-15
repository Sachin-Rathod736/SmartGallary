#!/usr/bin/env python
"""
CLIP Fine-tuning Script for AI Smart Gallery
Fine-tunes OpenCLIP ViT-B-32 on your photo collection for better image-text matching.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from pathlib import Path
import os

class PhotoDataset(Dataset):
    """Dataset for photos with captions"""
    def __init__(self, photo_dir, preprocess, tokenizer):
        self.photo_dir = Path(photo_dir)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.photos = list(self.photo_dir.glob('*.jpg')) + list(self.photo_dir.glob('*.png'))

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        img_path = self.photos[idx]
        image = self.preprocess(Image.open(img_path).convert('RGB'))

        # Use filename as caption (you can create a captions.txt file for better captions)
        caption = img_path.stem.replace('_', ' ')
        text = self.tokenizer([caption])

        return image, text.squeeze(0)

def train_clip(photo_dir, epochs=10, batch_size=32, lr=1e-5):
    """Fine-tune CLIP on your photos"""

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Create dataset
    dataset = PhotoDataset(photo_dir, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Training on {len(dataset)} photos for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0
        for images, texts in dataloader:
            images, texts = images.to(device), texts.to(device)

            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # Labels are diagonal (each image matches its text)
            labels = torch.arange(len(images), device=device)

            # Compute loss
            loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_clip.pth')
    print("Model saved as fine_tuned_clip.pth")

if __name__ == "__main__":
    # Usage: python train_clip.py
    photo_dir = "media/photos"  # Your photo directory
    train_clip(photo_dir)