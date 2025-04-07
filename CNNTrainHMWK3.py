import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters - Adjusted for higher accuracy
num_epochs = 25  # Increased from 20
batch_size = 64  # Reduced from 100
learning_rate = 0.003  # Increased from 0.002
num_experiments = 10
weight_decay = 1e-4  # Added L2 regularization

# Enhanced data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets with enhanced augmentation
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
train_size = int(0.8 * len(train_dataset))
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])

val_dataset.dataset.transform = test_transform 

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class EnhancedConvNet(nn.Module):
    def __init__(self):
        super(EnhancedConvNet, self).__init__()
        
        def conv_block(in_channels, out_channels, dropout_rate=0.1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate)
            )
        
        self.layer1 = conv_block(3, 32)   # Increased from 16
        self.layer2 = conv_block(32, 64)  # Increased from 32
        self.layer3 = conv_block(64, 128) # Increased from 64
        
        # Add a fourth layer for more capacity
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.25
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_model(model, data_loader, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_sum += loss.item()
    
    return correct / total, loss_sum / len(data_loader)

def run_experiment(experiment_num):
    model = EnhancedConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Initialize histories
    histories = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': []
    }
    
    print(f"Experiment {experiment_num+1}/{num_experiments} - Training started...")
    
    for epoch in range(num_epochs):
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        # Calculate and save epoch metrics
        epoch_train_acc = train_correct / train_total
        epoch_train_loss = train_loss / len(train_loader)
        histories['train_acc'].append(epoch_train_acc)
        histories['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        epoch_val_acc, epoch_val_loss = evaluate_model(model, val_loader, criterion)
        histories['val_acc'].append(epoch_val_acc)
        histories['val_loss'].append(epoch_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
    
    # Test the model
    test_accuracy, test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Experiment {experiment_num+1} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    
    # Add test results to histories
    histories['test_accuracy'] = test_accuracy
    histories['test_loss'] = test_loss
    
    return histories

# Plot accuracy metrics
def plot_metrics(metric_type, train_data, val_data, epochs, experiment_nums):
    plt.figure(figsize=(12, 5))
    
    mean_train = np.mean(train_data, axis=0)
    std_train = np.std(train_data, axis=0)
    mean_val = np.mean(val_data, axis=0)
    std_val = np.std(val_data, axis=0)
    
    if metric_type == 'Accuracy':
        plt.plot(epochs, mean_train, 'b-', label='Avg Train Acc')
        plt.plot(epochs, mean_val, 'r-', label='Avg Test Acc')
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, color='b', alpha=0.2, label='Train Acc Std Dev')
        plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, color='r', alpha=0.2, label='Test Acc Std Dev')
        plt.title('CNN Model - Average Training and Test Accuracy with Standard Deviation')
        
        plt.annotate(f'Final Train Acc: {mean_train[-1]:.4f}', 
                    xy=(epochs[-1], mean_train[-1]), 
                    xytext=(epochs[-4], mean_train[-1]+0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.annotate(f'Final Test Acc: {mean_val[-1]:.4f}', 
                    xy=(epochs[-1], mean_val[-1]), 
                    xytext=(epochs[-4], mean_val[-1]-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    else:
        plt.plot(epochs, mean_train, 'b-', label='Avg Train Loss')
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                        color='b', alpha=0.2, label='Train Loss Std Dev')
        plt.title('CNN Model - Average Training Loss with Standard Deviation')
        plt.annotate(f'Final Loss: {mean_train[-1]:.4f}', 
                    xy=(epochs[-1], mean_train[-1]), 
                    xytext=(epochs[-4], mean_train[-1]+0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_type)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if metric_type == 'Accuracy':
        plt.savefig('training_validation_curves.png')
    else:
        plt.savefig('training_loss_curve.png')

    plt.show()

if __name__ == '__main__':
    all_experiments = []
    test_metrics = {'accuracies': [], 'losses': []}

    for i in range(num_experiments):
        results = run_experiment(i)
        all_experiments.append(results)
        test_metrics['accuracies'].append(results['test_accuracy'])
        test_metrics['losses'].append(results['test_loss'])

    avg_test_accuracy = np.mean(test_metrics['accuracies'])
    std_test_accuracy = np.std(test_metrics['accuracies'])
    avg_test_loss = np.mean(test_metrics['losses'])
    std_test_loss = np.std(test_metrics['losses'])

    print(f"\nFinal Results: Average Test Accuracy: {avg_test_accuracy:.4f} ± {std_test_accuracy:.4f}, "
          f"Average Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")

    max_epochs = max(len(exp['train_acc']) for exp in all_experiments)
    train_acc_histories = np.array([exp['train_acc'] + [exp['train_acc'][-1]]*(max_epochs-len(exp['train_acc'])) for exp in all_experiments])
    val_acc_histories = np.array([exp['val_acc'] + [exp['val_acc'][-1]]*(max_epochs-len(exp['val_acc'])) for exp in all_experiments])
    train_loss_histories = np.array([exp['train_loss'] + [exp['train_loss'][-1]]*(max_epochs-len(exp['train_loss'])) for exp in all_experiments])
    val_loss_histories = np.array([exp['val_loss'] + [exp['val_loss'][-1]]*(max_epochs-len(exp['val_loss'])) for exp in all_experiments])

    epochs = range(1, max_epochs + 1)
    experiment_nums = range(1, num_experiments + 1)

    plot_metrics('Accuracy', train_acc_histories, val_acc_histories, epochs, experiment_nums)
    plot_metrics('Loss', train_loss_histories, val_loss_histories, epochs, experiment_nums)
    