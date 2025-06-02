
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim

# ------------------------------
# Configuration
# ------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps") # For MacBook M Chip
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
num_classes = 10
batch_size = 128
T = 2.0
alpha = 0.5
lr_weights = 0.01
lr_IW = 0.0001
lr_IX = 0.01
epochs_SS = 1 # Original: 30 for CIFAR-10
epochs_CS = 1 # Original: 100 for CIFAR-10
epochs_TU = 1 # Original: 70 for CIFAR-10

# ------------------------------
# Data Preparation
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# Models
# ------------------------------
def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class QuantizedModel(nn.Module):
    def __init__(self, base_model, bit_width=4):
        super().__init__()
        self.model = base_model
        self.bit_width = bit_width
        self.IW = nn.Parameter(torch.tensor(1.0))
        self.IX = nn.Parameter(torch.tensor(1.0))

    def quantize_weights(self, w):
        qmin = -2 ** (self.bit_width - 1)
        qmax = 2 ** (self.bit_width - 1) - 1
        w_clamped = torch.clamp(w / self.IW, qmin, qmax)
        w_rounded = torch.round(w_clamped) * self.IW
        return w_rounded

    def quantize_activations(self, x):
        qmin = 0
        qmax = 2 ** self.bit_width - 1
        x_clamped = torch.clamp(x / self.IX, qmin, qmax)
        x_rounded = torch.round(x_clamped) * self.IX
        return x_rounded

    def forward(self, x):
        x = self.quantize_activations(self.model.conv1(x))
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        self.model.fc.weight.data = self.quantize_weights(self.model.fc.weight.data)
        x = self.model.fc(x)
        return x

# ------------------------------
# Loss Functions
# ------------------------------
def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def distillation_loss(student_logits, teacher_logits, target):
    ce = F.cross_entropy(student_logits, target)
    kl = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                  F.softmax(teacher_logits / T, dim=1),
                  reduction='batchmean') * (T * T)
    return alpha * kl + (1. - alpha) * ce

# ------------------------------
# Training Phases
# ------------------------------
def train_self_study(student, loader):
    student.train()
    optimizer = optim.SGD([
        {'params': student.model.parameters(), 'lr': lr_weights},
        {'params': [student.IW], 'lr': lr_IW},
        {'params': [student.IX], 'lr': lr_IX}
    ])
    for epoch in range(epochs_SS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = student(x)
            loss = cross_entropy_loss(output, y)
            loss.backward()
            optimizer.step()
        print("\t Epoch:{} Loss: {}".format(epoch,loss))

def train_co_study(student, teacher, loader):
    student.train()
    teacher.train()
    optimizer_s = optim.SGD([
        {'params': student.model.parameters(), 'lr': lr_weights},
        {'params': [student.IW], 'lr': lr_IW},
        {'params': [student.IX], 'lr': lr_IX}
    ])
    optimizer_t = optim.SGD(teacher.parameters(), lr=lr_weights)
    for epoch in range(epochs_CS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Teacher update
            t_output = teacher(x)
            s_output = student(x).detach()
            t_loss = distillation_loss(t_output, s_output, y)
            optimizer_t.zero_grad()
            t_loss.backward()
            optimizer_t.step()
            # Student update
            with torch.no_grad():
                t_output = teacher(x)
            s_output = student(x)
            s_loss = distillation_loss(s_output, t_output, y)
            optimizer_s.zero_grad()
            s_loss.backward()
            optimizer_s.step()
        print("\t Epoch:{} Loss: {}".format(epoch,s_loss))

def train_tutoring(student, teacher, loader):
    student.train()
    teacher.eval()
    optimizer = optim.SGD([
        {'params': student.model.parameters(), 'lr': lr_weights},
        {'params': [student.IW], 'lr': lr_IW},
        {'params': [student.IX], 'lr': lr_IX}
    ])
    for epoch in range(epochs_TU):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_output = teacher(x)
            s_output = student(x)
            loss = distillation_loss(s_output, t_output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("\t Epoch:{} Loss: {}".format(epoch,loss))


# ------------------------------
# Evaluation
# ------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = 100. * correct / total
    print(f"** Accuracy: {acc:.2f}%")
    return acc

def main():
    teacher = create_model().to(device)
    student_fp = create_model().to(device)
    student = QuantizedModel(student_fp).to(device)

    print("[Phase 1] Self-Studying")
    train_self_study(student, train_loader)
    evaluate(student, train_loader)

    print("[Phase 2] Co-Studying")
    train_co_study(student, teacher, train_loader)
    evaluate(student, train_loader)

    print("[Phase 3] Tutoring")
    train_tutoring(student, teacher, train_loader)
    evaluate(student, train_loader)

    torch.save(student.state_dict(), "qkd_final_student.pth")

if __name__ == '__main__':
    main()
