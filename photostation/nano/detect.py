import os

from PIL import Image
import cv2

import config

if config.with_mtcnn:
    from facenet_pytorch import MTCNN
    from PIL import ImageDraw, Image
    import torch
    import torch.nn.functional as F
    import numpy as np
    from torchvision import datasets, transforms, models


def setup_model():
    global g_device, g_mtcnn, g_categories, g_trans, g_model

    # For live detection
    if config.with_mtcnn:
        g_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        g_mtcnn = MTCNN(keep_all=True, device=g_device)
    
        data_dir = os.path.join(
            config.mount_root_dir, 
            config.dataset_used_for_training
        )
        model_name = config.model_name
        model_path_value = config.model_path_value
        dataset = datasets.ImageFolder(
            data_dir, 
            transform=transforms.Resize((224, 224)))
       
        if config.model_type == 'resnet18':
            g_model = models.resnet18(pretrained=True)
            g_model.fc = torch.nn.Linear(512, len(dataset.class_to_idx))
        if config.model_type == 'vgg11':
            g_model = models.vgg11(pretrained=True)
            g_model.fc = torch.nn.Linear(512, len(dataset.class_to_idx))
    
        custom_model_path = os.path.join(model_path_value, f'{model_name}.pth')
        g_model.load_state_dict(torch.load(custom_model_path))
        g_model.to(g_device)
        g_model = g_model.eval()
        g_categories = dict([(i[1], i[0]) for i in dataset.class_to_idx.items()])
        
        g_trans = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def apply_on_image(image_full_path):
    detected_boxes = []
    image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    boxes, _ = g_mtcnn.detect(image)
    detected, prob = g_mtcnn(image, return_prob=True) 
    if detected is not None:
        print(f'Face detected with probability: {prob}')
        detected = [detected]
        detected = torch.stack(detected).to(g_device)
        output = g_model(detected[0]) #.detach().cpu() #.numpy().flatten()
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        if category_index <= len(g_categories):
            print(f'{g_categories[category_index]}')
        else:
            print(f'out of categories {category_index}')
    if boxes is not None:
        for box in boxes:
            _box = box.tolist()
            print(_box)
            detected_box = {
                'x': _box[0]/width,
                'y': _box[1]/height,
                'w': (_box[2] - _box[0])/width,
                'h': (_box[3] - _box[1])/height,
                'cat': ''
            }
            if category_index in g_categories.keys():
                detected_box['cat'] = g_categories[category_index]
            detected_boxes.append(detected_box)
    return detected_boxes

setup_model()
