import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from collections import Counter

folder_path_train = "traintestfolder/train/images"
folder_path_captions_train = "traintestfolder/train/captions.txt"
batch_size = 32
num_epochs = 10
learning_rate = 0.001
embed_size = 256
hidden_size = 256
num_layers = 1
min_word_freq = 5

def load_image_caption_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().split(",", maxsplit=1) for line in f.readlines()]

def build_vocab(image_caption_pairs, min_freq):
    word_counts = Counter()
    for _, caption in image_caption_pairs:
        word_counts.update(caption.lower().split())
    vocab = ["<pad>", "<sos>", "<eos>", "<unk>"]
    vocab.extend([w for w, c in word_counts.items() if c >= min_freq])
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return word_to_idx, vocab

class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, captions_list, word_to_idx):
        self.image_folder = image_folder
        self.captions_list = captions_list
        self.word_to_idx = word_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.captions_list)

    def __getitem__(self, idx):
        image_name, caption = self.captions_list[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = self.transform(Image.open(image_path).convert("RGB"))
        caption_tokens = ["<sos>"] + caption.lower().split() + ["<eos>"]
        token_indices = [self.word_to_idx.get(t, self.word_to_idx["<unk>"]) for t in caption_tokens]
        return image, torch.tensor(token_indices)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded

class ResNetEncoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in resnet.parameters():
            p.requires_grad_(False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, output_size)
        self.resnet = resnet
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, images):
        return self.bn(self.resnet(images))

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        return self.fc_out(outputs)

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions_input):
        features = self.encoder(images)
        return self.decoder(features, captions_input)

image_caption_pairs_train = load_image_caption_pairs(folder_path_captions_train)
word_to_idx, vocab = build_vocab(image_caption_pairs_train, min_freq=min_word_freq)
vocab_size = len(vocab)
pad_idx = word_to_idx["<pad>"]

dataset_train = ImageCaptionDataset(folder_path_train, image_caption_pairs_train, word_to_idx)
loader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ResNetEncoder(hidden_size).to(device)
decoder = LSTMDecoder(vocab_size, embed_size, hidden_size, num_layers).to(device)
model = EncoderDecoderModel(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, captions) in enumerate(loader_train):
        images, captions = images.to(device), captions.to(device)
        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]
        optimizer.zero_grad()
        outputs = model(images, decoder_input)
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(loader_train)}] Loss: {loss.item():.4f}")
    avg_epoch_loss = total_loss / len(loader_train)
    print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_epoch_loss:.4f}")
