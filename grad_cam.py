import torch
from torch import nn
from torchvision import transforms, datasets, models
import cv2
import numpy as np
from MyDataset import MyDataset

class Cam(nn.Module):
    def __init__(self, net, layerName):
        super(Cam, self).__init__()
        self.fp = None
        self.grad = None
        self.net = net
        self.register_hook(layerName)

    '''获得网络中间层的输出'''
    def _get_feature_hook(self, module, input, out):
        self.fp = out
    '''获得中间梯度'''
    def _get_grad_hook(self, module, input, out):
        print(len(out))
        self.grad = out[0]
        print(self.grad.shape)

    def register_hook(self, layerName):
        self.net._modules.get(layerName).register_forward_hook(self._get_feature_hook)
        self.net._modules.get(layerName).register_backward_hook(self._get_grad_hook)

    def forward(self, img, label=None):
        out = self.net(img)
        if not label:
            label = out.argmax(dim=1).int().item()
        out[0, label].backward()
        w = nn.functional.adaptive_avg_pool2d(self.grad, (1,1))
        '''网络分类层的名称为fc'''
        # w = self.net.fc.weight[label]
        # w = w.view(-1, 1, 1)
        cam = w * self.fp
        cam = cam.sum(dim=1)
        cam = cam.detach().squeeze(0).unsqueeze(2).numpy()
        '''数值归一化'''
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam_img = np.uint8(cam*255)
        '''生成伪彩图'''
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (256, 256)), 2)
        img = denormalize(img, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        img = img.permute(0,2,3,1).squeeze(0).numpy()
        img = np.uint8(img * 255)
        heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
        '''返回伪彩图与其叠加图'''
        return heatmap, 0.5 * heatmap + 0.3 * img

'''逆归一化'''
def denormalize(img, mean, std):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = img * std
    img = img + mean
    return img

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

dataset = MyDataset('./train', transform)
net = models.resnet18()
net.fc = nn.Linear(512, 2)
net.load_state_dict(torch.load("./resnet18_2.pkl"))
cam = Cam(net, 'layer4')
img, label = dataset[10]
heatmap, res = cam(img.unsqueeze(0), label)
img = denormalize(img, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
img = img.squeeze(0).permute(1, 2, 0).numpy()
img = np.uint8(img * 255)
cv2.imwrite("grad_raw.jpg", img)
cv2.imwrite("grad_heatmap.jpg", heatmap)
cv2.imwrite("grad_res.jpg", res)


