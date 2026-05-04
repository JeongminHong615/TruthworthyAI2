import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

def get_inverse_norm(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(-1, 1, 1).to(device)
    x_clone = x.clone() * std + mean
    return torch.clamp(x_clone, 0, 1)

def save_disagreement(img_tensor, pred1, pred2, sample_idx):
    if not os.path.exists('results'):
        os.makedirs('results')
        
    img_disp = img_tensor.cpu().squeeze().numpy()
    img_disp = np.transpose(img_disp, (1, 2, 0)) 
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img_disp)
    plt.title(f"Disagreement Detected!\nModel 1 Pred: {pred1} | Model 2 Pred: {pred2}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"results/disagreement_sample_{sample_idx+1}.png")
    plt.close()

def main():
    # CIFAR-10
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # ResNet50
    print("Loading Model 1 (Pre-trained on CIFAR-10)")
    model1 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model1.fc = torch.nn.Linear(model1.fc.in_features, 10) # CIFAR-10 클래스(10)에 맞게 수정
    model1 = model1.to(device)
    model1.eval()

    print("Loading Model 2 (Different Initialization)")
    model2 = torchvision.models.resnet50(weights=None) # 아예 랜덤 초기화된 가중치 사용
    model2.fc = torch.nn.Linear(model2.fc.in_features, 10)
    model2 = model2.to(device)
    model2.eval()

    coverage_threshold = 0.5
    activated_neurons = set()
    total_neurons = 0

    # Hook 함수
    def hook_fn(module, input, output):
        nonlocal total_neurons
        activations = output.detach().cpu().numpy()
        if total_neurons == 0:
            total_neurons = activations.size // activations.shape[0]
        
        flattened = activations.reshape(activations.shape[0], -1)
        for i in range(flattened.shape[0]):
            active_idx = np.where(flattened[i] > coverage_threshold)[0]
            activated_neurons.update(active_idx.tolist())

    model1.layer4.register_forward_hook(hook_fn)

    # Differential Testing
    print("\nRunning Differential Testing (DeepXplore style)")
    disagreements = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            
            out1 = model1(data)
            out2 = model2(data)
            
            pred1 = out1.argmax(dim=1).item()
            pred2 = out2.argmax(dim=1).item()
            
            if pred1 != pred2:
                data_01 = get_inverse_norm(data[0])
                save_disagreement(data_01, pred1, pred2, disagreements)
                disagreements += 1
                print(f"    - Disagreement found! ({disagreements}/5) : M1={pred1}, M2={pred2}")
                
            if disagreements >= 5: 
                break

    coverage = len(activated_neurons) / total_neurons if total_neurons > 0 else 0
    print(f"Differential Testing Completed!")
    print(f"    - Total Disagreement Inputs Saved: {disagreements}/5")
    print(f"    - Achieved Neuron Coverage: {coverage * 100:.2f}%")

if __name__ == '__main__':
    main()