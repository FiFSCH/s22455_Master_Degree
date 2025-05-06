import torch
import torch.nn as nn
from torchvision import transforms, datasets
from Auxilary.Live_camera_footage_capture import capture_camera_footage


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = datasets.ImageFolder(root='../PSL_Collected_Data', transform=transform).classes
number_of_classes = len(class_names)

# Load og model, adjust it custom task and load saved version of the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, number_of_classes)
)
model.load_state_dict(torch.load('mobileNetV2_psl.pth', map_location=torch.device('cpu')))

# Switch to evaluation mode
model.eval()

capture_camera_footage(transform, model, class_names)