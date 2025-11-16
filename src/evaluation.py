import torch
import os
from tqdm import tqdm

def inference(model, dataloader, label_map, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    total = 0

    index_to_label = {idx: label for label, idx in label_map.items()}
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Inference", leave=False)
        for images, labels, image_paths in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            filenames = [os.path.basename(path) for path in image_paths]
            all_images.extend(filenames)
            all_preds.extend([index_to_label[int(idx)] for idx in predicted.cpu().numpy()])
            all_labels.extend([index_to_label[int(idx)] for idx in labels.cpu().numpy()])
            total += labels.size(0)
    
            progress_bar.set_postfix(total_samples=total)
        
    return all_images, all_preds, all_labels