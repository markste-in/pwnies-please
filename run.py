import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torchvision import transforms

pretrained_model = "pwny_cifar_eps_0.pth"
lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imagenet_class_index = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.255])])

def attack(model, device, test_loader, lr):
    hit = False
    data, target = test_loader
    data.requires_grad = True
    data, target = data.requires_grad_(True).to(device), target.to(device)

    while not hit:
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        sign_data_grad = data.grad.data.sign()
        data.data = torch.clamp(data - lr * sign_data_grad,0,1)
        output = model(data)
        initial_category = output.max(1, keepdim=True)[1].item()
        print("Class", imagenet_class_index[initial_category],"# Loss",loss.detach().numpy())
        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            hit = True
            img = inv_normalize(data.detach().squeeze()).numpy().swapaxes(0,1).swapaxes(1,2) * 255
            Image.fromarray(img.astype(np.uint8)).save("collision.png")
    return img

image = Image.open("download.png")
image = np.array(image)

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(imagenet_class_index))
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model_ft = model.to("cpu")
model.eval()

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)
dl = torch.utils.data.DataLoader((input_tensor,2))

result = attack(model, device, dl, lr)