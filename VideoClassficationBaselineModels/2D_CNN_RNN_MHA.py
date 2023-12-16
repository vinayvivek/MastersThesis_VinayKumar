import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms for the input videos
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the UCF101 dataset
train_dataset = datasets.UCF101('/home/multi-sy-20/PycharmProjects/MS_SSCL_VC_AR_VinayKumar/data/UCF-101', 
                                annotation_path='/home/multi-sy-20/PycharmProjects/MS_SSCL_VC_AR_VinayKumar/data/ucfTrainTestlist/trainlist03_updated.txt', 
                                frames_per_clip=64, step_between_clips=1, transform=transform, is_val=False)

val_dataset = datasets.UCF101('/home/multi-sy-20/PycharmProjects/MS_SSCL_VC_AR_VinayKumar/data/UCF-101', 
                              annotation_path='/home/multi-sy-20/PycharmProjects/MS_SSCL_VC_AR_VinayKumar/data/ucfTrainTestlist/testlist03_updated.txt', 
                              frames_per_clip=64, step_between_clips=1, transform=transform, is_val=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define 2D ResNet18 model
backbone = torchvision.models.resnet18(pretrained=True)
backbone.fc = torch.nn.Identity()  # Remove the classification layer

# Define the RNN (LSTM) module with multi-headed attention
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = torch.nn.functional.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.fc_out(x)
        
        return x

class CNNRNNAttention(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, num_heads=8):
        super(CNNRNNAttention, self).__init__()
        self.backbone = backbone
        self.rnn = nn.LSTM(input_size=backbone.fc.in_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.attention = MultiHeadAttention(input_size=hidden_size, hidden_size=hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.backbone(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        
        r_out, _ = self.rnn(r_in)
        attended = self.attention(r_out, r_out, r_out)
        
        out = self.fc(attended[:, -1, :])  # Taking the last timestep's output
        return out

# Define the CNN-RNN model with multi-headed attention
num_classes = len(train_dataset.classes)
hidden_size = 512  # Adjust according to your needs
num_heads = 8  # Number of attention heads
model = CNNRNNAttention(backbone, hidden_size, num_classes, num_heads)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
