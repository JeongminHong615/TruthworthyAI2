import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}\n")

# MNIST
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # dataload
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = CNN().to(device)
    model_path = "mnist_cnn.pth"
    
    # 가중치 파일이 있으면 로드, 없으면 새로 학습
    if os.path.exists(model_path):
        print("[*] 기존에 학습된 MNIST 모델 가중치를 불러오기")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("[*] 학습 시작")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        epochs = 3
        for epoch in range(1, epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f"    - Epoch {epoch}/{epochs} 학습 완료")
            
        torch.save(model.state_dict(), model_path)
        print(f"[*] MNIST 모델 학습 및 저장 완료: {model_path}")
        
    return model, test_loader

# CIFAR-10
def get_cifar():
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet50", pretrained=True)
    model = model.to(device)
    model2 = torchvision.models.resnet50(num_classes=10)
    model2 = model2.to(device)

    
    return model, test_loader

# accuracy
def test_model(model, device, test_loader, dataset_name):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f" {dataset_name} Clean Test Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy

if __name__ == '__main__':
    mnist_model, mnist_test_loader = get_mnist()
    test_model(mnist_model, device, mnist_test_loader, "MNIST")
    
    cifar_model, cifar_test_loader = get_cifar()
    test_model(cifar_model, device, cifar_test_loader, "CIFAR-10")


def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    x_adv.retain_grad()

    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, target)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv -= eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    x_adv.retain_grad()

    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, label)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv += eps * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def pgd_targeted(model, x, target, k, eps, eps_step):
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv = x_adv.clone().detach()
        x_adv.requires_grad = True
        x_adv.retain_grad()

        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, target)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv -= eps_step * x_adv.grad.sign()
            eta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = x + eta
            x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_adv = x.clone().detach()
    
    for _ in range(k):
        x_adv = x_adv.clone().detach()
        x_adv.requires_grad = True
        x_adv.retain_grad()
        
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, label)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + eps_step * x_adv.grad.sign()
            eta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + eta, min=0, max=1)
            
    return x_adv.detach()

# regularization
class AttackWrapper(nn.Module): # attack 함수가 모델 내부에서만 0~1범위로 정규화하게
    def __init__(self, model, dataset_name):
        super().__init__()
        self.model = model
        if dataset_name == "MNIST":
            self.mean = torch.tensor([0.1307]).view(-1, 1, 1).to(device)
            self.std = torch.tensor([0.3081]).view(-1, 1, 1).to(device)
        else:
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(device)
            self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(-1, 1, 1).to(device)
            
    def forward(self, x): # 0~1 범위의 x를 모델에게 맞게 정규화
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)

def get_inverse_norm(x, dataset_name): # 정규화된 이미지를 [0,1] 범위로 돌림
    x_clone = x.clone()
    if dataset_name == "MNIST":
        x_clone = x_clone * 0.3081 + 0.1307
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(-1, 1, 1).to(device)
        x_clone = x_clone * std + mean
    return torch.clamp(x_clone, 0, 1)

# visualization
def save_visualization(clean_img, adv_img, clean_pred, adv_pred, eps, attack_name, dataset_name, sample_idx):
    # 결과를 원본, 조작본, 노이즈 형태로 반환
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # 텐서 -> numpy
    clean_disp = clean_img.cpu().squeeze().numpy()
    adv_disp = adv_img.cpu().squeeze().numpy()
    if dataset_name == "CIFAR-10":
        clean_disp = np.transpose(clean_disp, (1, 2, 0))
        adv_disp = np.transpose(adv_disp, (1, 2, 0))
        
    # 노이즈 계산
    noise = (adv_img - clean_img).cpu().squeeze().numpy()
    if dataset_name == "CIFAR-10":
        noise = np.transpose(noise, (1, 2, 0))
    noise_disp = np.clip(noise * 5.0 + 0.5, 0, 1) # 노이즈 변화량을 증폭시킴

    plt.figure(figsize=(10, 3))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(clean_disp, cmap='gray' if dataset_name == "MNIST" else None)
    plt.title(f"Original (Pred: {clean_pred})")
    plt.axis('off')
    
    # 조작 이미지
    plt.subplot(1, 3, 2)
    plt.imshow(adv_disp, cmap='gray' if dataset_name == "MNIST" else None)
    plt.title(f"Adversarial (Pred: {adv_pred})")
    plt.axis('off')
    
    # 노이즈
    plt.subplot(1, 3, 3)
    plt.imshow(noise_disp, cmap='gray' if dataset_name == "MNIST" else None)
    plt.title(f"Perturbation (eps={eps})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_{attack_name}_eps{eps}_sample{sample_idx}.png")
    plt.close()

# evaluation
def run_evaluation(model, test_loader, attack_fn, attack_name, dataset_name, eps, is_targeted):
    model.eval()
    success_count = 0
    total_count = 0
    saved_images = 0
    MAX_SAMPLES = 100 
    
    for data, target in test_loader:
        if total_count >= MAX_SAMPLES:
            break
            
        data, target = data.to(device), target.to(device)
        data_01 = get_inverse_norm(data, dataset_name) # 정규화된 이미지를 다시 원본 이미지로
        
        # 모델의 원래 이미지 예측
        clean_output = model(data_01)
        clean_pred = clean_output.argmax(dim=1, keepdim=True).squeeze()
        
        # 정답을 맞춘 이미지만 공격
        correct_idx = (clean_pred == target).nonzero(as_tuple=False).squeeze()
        if correct_idx.dim() == 0: correct_idx = correct_idx.unsqueeze(0)
        if len(correct_idx) == 0: continue
            
        data_01, target, clean_pred = data_01[correct_idx], target[correct_idx], clean_pred[correct_idx]
        
        # 타겟 공격일때 가짜 라벨 생성
        target_labels = (target + 1) % 10 if is_targeted else target
        
        # 공격
        if "pgd" in attack_name.lower():
            k_steps = 40 if dataset_name == "MNIST" else 20
            eps_step = 0.01 if dataset_name == "MNIST" else eps / 10
            adv_data = attack_fn(model, data_01, target_labels, k=k_steps, eps=eps, eps_step=eps_step)
        else:
            adv_data = attack_fn(model, data_01, target_labels, eps=eps)
            
        # 조작된 이미지에 대한 예측
        adv_output = model(adv_data)
        adv_pred = adv_output.argmax(dim=1, keepdim=True).squeeze()
        
        for i in range(len(target)):
            is_success = (adv_pred[i] == target_labels[i]) if is_targeted else (adv_pred[i] != target[i]) 
            if is_success:
                success_count += 1
            total_count += 1
            
            if is_success and saved_images < 5:
                saved_images += 1
                save_visualization(
                    data_01[i:i+1], adv_data[i:i+1], 
                    clean_pred[i].item(), adv_pred[i].item(), 
                    eps, attack_name, dataset_name, saved_images
                )
                
        if total_count >= MAX_SAMPLES:
            break

    # 최종 공격 성공률 계산
    success_rate = 100. * success_count / total_count
    print(f"[{dataset_name}] {attack_name} (eps={eps}) -> Success Rate: {success_count}/{total_count} ({success_rate:.2f}%)")
    return success_rate


if __name__ == '__main__':
    # dataload
    mnist_model, mnist_test_loader = get_mnist()
    cifar_model, cifar_test_loader = get_cifar()
    
    print("베이스라인 성능 확인")
    test_model(mnist_model, device, mnist_test_loader, "MNIST")
    test_model(cifar_model, device, cifar_test_loader, "CIFAR-10")
    print("="*50)
    
    # 공격 설정
    epsilons = [0.05, 0.1, 0.2, 0.3]
    attacks = [
        ("Targeted_FGSM", fgsm_targeted, True),
        ("Untargeted_FGSM", fgsm_untargeted, False),
        ("Targeted_PGD", pgd_targeted, True),
        ("Untargeted_PGD", pgd_untargeted, False)
    ]
    
    datasets = [
        ("MNIST", mnist_model, mnist_test_loader),
        ("CIFAR-10", cifar_model, cifar_test_loader)
    ]
        
    for dataset_name, model, test_loader in datasets:
        print(f"\n{'='*15} {dataset_name} 공격 시작 {'='*15}")
        wrapped_model = AttackWrapper(model, dataset_name).to(device)
        
        for attack_name, attack_fn, is_targeted in attacks:
            print(f"\n--- {attack_name} ---")
            for eps in epsilons:
                run_evaluation(wrapped_model, test_loader, attack_fn, attack_name, dataset_name, eps, is_targeted)