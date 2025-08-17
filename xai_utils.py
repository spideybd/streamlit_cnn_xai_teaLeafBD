import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from lime import lime_image
from skimage.segmentation import mark_boundaries

transform_for_lime = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalize_image(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normalize(tensor)

def transformed_img(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.dtype in [np.float32, np.float64]:
        img = np.clip(img, 0, 1)
    return img

def get_target_layers(model, model_name):
    if model_name == "CustomCNN":
        return [model.features[-2]]
    elif model_name == "ResNet152":
        return [model.layer4[-1]]
    elif model_name == "DenseNet201":
        return [model.features.denseblock4.denselayer32]
    elif model_name == "EfficientNetB3":
        return [model.features[-1][0]]
    else:
        raise ValueError("Model architecture not supported for CAM.")

def generate_cam_visualizations(model, model_name, input_tensor, original_image):
    target_layers = get_target_layers(model, model_name)
    visualizations = {}
    
    vis_image = np.array(original_image.resize((224, 224))) / 255.0
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualizations["Grad-CAM"] = transformed_img(
            show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
        )

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualizations["Grad-CAM++"] = transformed_img(
            show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
        )
    
    with EigenCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualizations["Eigen-CAM"] = transformed_img(
            show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
        )

    with AblationCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualizations["Ablation-CAM"] = transformed_img(
            show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
        )

    return visualizations

def generate_lime_visualization(model, image_pil, device):
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        pil_imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        tensor_batch = torch.stack([transform_for_lime(p) for p in pil_imgs]).to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(tensor_batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explanation = explainer.explain_instance(
        np.array(image_pil.resize((224, 224))),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    return transformed_img(lime_img)
