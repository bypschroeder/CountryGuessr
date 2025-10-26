import torch
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device, writer, global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
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

        writer.add_scalar("Loss/Train", batch_loss, global_step)
        writer.add_scalar("Accuracy/Train", batch_acc, global_step)
        global_step += 1

        progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, writer, global_step):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_loss = running_loss / total
            batch_acc = correct / total

            writer.add_scalar("Loss/Val", batch_loss, global_step)
            writer.add_scalar("Accuracy/Val", batch_acc, global_step)
            global_step += 1

            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
