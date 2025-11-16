import torch
import os
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device, writer, global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, image_paths in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        batch_loss = running_loss / total
        batch_acc = correct / total

        writer.add_scalar("Train/Loss", batch_loss, global_step)
        writer.add_scalar("Train/Accuracy", batch_acc, global_step)
        global_step += 1

        progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, global_step


def validate_epoch(model, dataloader, criterion, device, writer, global_step):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels, image_paths in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_loss = running_loss / total
            batch_acc = correct / total

            writer.add_scalar("Validation/Loss", batch_loss, global_step)
            writer.add_scalar("Validation/Accuracy", batch_acc, global_step)
            global_step += 1

            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, all_preds, all_labels, global_step


def save_checkpoint(model, optimizer, epoch, label_map, train_loss=None, val_loss=None, train_acc=None, val_acc=None, seed=None, checkpoint_dir="model", filename=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"epoch_{epoch}.pth"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "label_map": label_map
    }
    
    # Optionally add metrics and seed if provided
    if train_loss is not None:
        checkpoint['train_loss'] = train_loss
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    if train_acc is not None:
        checkpoint['train_acc'] = train_acc
    if val_acc is not None:
        checkpoint['val_acc'] = val_acc
    if seed is not None:
        checkpoint['seed'] = seed
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
