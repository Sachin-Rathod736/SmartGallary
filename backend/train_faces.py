#!/usr/bin/env python
"""
Face Recognition Training Script
Trains a simple classifier on face embeddings for person identification.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
import os

class FaceEmbeddingDataset(Dataset):
    """Dataset of face embeddings with person labels"""
    def __init__(self, embedding_file):
        # Load embeddings and labels from a file
        # Format: each line "person_name,embedding_values"
        self.embeddings = []
        self.labels = []

        with open(embedding_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                person = parts[0]
                embedding = [float(x) for x in parts[1:]]

                self.labels.append(person)
                self.embeddings.append(embedding)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return embedding, label

class FaceClassifier(nn.Module):
    """Simple neural network for face classification"""
    def __init__(self, embedding_dim=512, num_classes=10):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def collect_face_embeddings(photo_dir, output_file):
    """Collect face embeddings from photos (requires InsightFace)"""
    try:
        from ai_engine.face_recognition.detector import detect_faces
        from gallery.models import Photo
        import django

        # Setup Django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
        django.setup()

        print("Collecting face embeddings...")

        with open(output_file, 'w') as f:
            photo_paths = list(Path(photo_dir).glob('*.jpg')) + list(Path(photo_dir).glob('*.png'))

            for photo_path in photo_paths:
                # Extract person name from filename (customize this logic)
                person_name = photo_path.stem.split('_')[0]  # Assumes format: person_image.jpg

                try:
                    detections = detect_faces(str(photo_path))
                    if detections:
                        # Use the first face
                        embedding = detections[0]['embedding']
                        embedding_str = ','.join(map(str, embedding))
                        f.write(f"{person_name},{embedding_str}\n")
                        print(f"Processed {photo_path.name}")
                except Exception as e:
                    print(f"Error processing {photo_path}: {e}")

        print(f"Embeddings saved to {output_file}")

    except ImportError:
        print("InsightFace not available. Install with: pip install insightface")

def train_face_classifier(embedding_file, epochs=50, batch_size=32, lr=1e-3):
    """Train face classifier"""

    # Load dataset
    dataset = FaceEmbeddingDataset(embedding_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    num_classes = len(dataset.label_encoder.classes_)
    model = FaceClassifier(num_classes=num_classes)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Training classifier for {num_classes} people on {len(dataset)} faces")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'face_classifier.pth')
    torch.save(dataset.label_encoder, 'label_encoder.pth')
    print("Model saved as face_classifier.pth")

if __name__ == "__main__":
    # Step 1: Collect embeddings (run this first)
    # collect_face_embeddings("media/photos", "face_embeddings.txt")

    # Step 2: Train classifier (run after collecting embeddings)
    # train_face_classifier("face_embeddings.txt")