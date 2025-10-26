#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models.EEGNet_my import EEGNet
from dataloader_my import read_bci_data
import pandas as pd

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, 
                       num_epochs, device):
    """訓練並評估模型"""
    best_acc = 0.0
    train_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 測試
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)
    
    return best_acc, train_losses, test_accs

def experiment_activation_functions():
    """實驗1: 不同激活函數"""
    print("\n" + "="*60)
    print("實驗1: 比較不同激活函數")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_label, test_data, test_label = read_bci_data()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data), torch.LongTensor(train_label))
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data), torch.LongTensor(test_label))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    activations = ['relu', 'leaky_relu', 'elu']
    results = {}
    
    for activation in activations:
        print(f"\n測試激活函數: {activation}")
        model = EEGNet(activation=activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_acc, train_losses, test_accs = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, 
            num_epochs=300, device=device)
        
        results[activation] = {
            'best_acc': best_acc,
            'train_losses': train_losses,
            'test_accs': test_accs
        }
        print(f"✅ {activation} 最佳準確率: {best_acc:.2f}%")
    
    # 繪製比較圖
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for activation in activations:
        plt.plot(results[activation]['train_losses'], label=activation)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for activation in activations:
        plt.plot(results[activation]['test_accs'], label=activation)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_activation_functions.png', dpi=300)
    print("\n✅ 圖表已儲存: experiment_activation_functions.png")
    
    return results


def experiment_learning_rates():
    """實驗2: 不同學習率"""
    print("\n" + "="*60)
    print("實驗2: 比較不同學習率")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_label, test_data, test_label = read_bci_data()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data), torch.LongTensor(train_label))
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data), torch.LongTensor(test_label))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        print(f"\n測試學習率: {lr}")
        model = EEGNet(activation='elu').to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_acc, train_losses, test_accs = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, 
            num_epochs=300, device=device)
        
        results[f'lr_{lr}'] = {
            'best_acc': best_acc,
            'train_losses': train_losses,
            'test_accs': test_accs
        }
        print(f"✅ 學習率 {lr} 最佳準確率: {best_acc:.2f}%")
    
    # 繪製比較圖
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for lr in learning_rates:
        plt.plot(results[f'lr_{lr}']['train_losses'], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Learning Rate Comparison - Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for lr in learning_rates:
        plt.plot(results[f'lr_{lr}']['test_accs'], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Learning Rate Comparison - Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_learning_rates.png', dpi=300)
    print("\n✅ 圖表已儲存: experiment_learning_rates.png")
    
    return results


if __name__ == '__main__':
    # 執行所有實驗
    results_activation = experiment_activation_functions()
    results_lr = experiment_learning_rates()

    
    # 生成實驗結果摘要
    print("\n" + "="*60)
    print("實驗結果摘要")
    print("="*60)
    
    print("\n激活函數比較:")
    for activation, data in results_activation.items():
        print(f"  {activation:15s}: {data['best_acc']:.2f}%")
    
    print("\n學習率比較:")
    for lr_name, data in results_lr.items():
        print(f"  {lr_name:15s}: {data['best_acc']:.2f}%")

# %%
