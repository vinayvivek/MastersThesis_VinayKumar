import torch
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

# Define the RNN (LSTM) module
class CNNLSTM(torch.nn.Module):
    def __init__(self, backbone, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()
        self.backbone = backbone
        self.lstm = torch.nn.LSTM(input_size=backbone.fc.in_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.backbone(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.lstm(r_in)
        r_out = self.fc(r_out[:, -1, :])  # Taking the last timestep's output
        return r_out

# Define the CNN-LSTM model
num_classes = len(train_dataset.classes)
hidden_size = 512  # Adjust according to your needs
model = CNNLSTM(backbone, hidden_size, num_classes)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
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
