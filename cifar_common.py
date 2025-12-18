import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def get_data(batch_size: int, num_workers: int = 4, prefetch_factor: int = 4, normalize_test: bool = True):
    """Load CIFAR-10 dataset with data augmentation.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches to prefetch per worker
        normalize_test: Whether to normalize test data (default: True)
    """
    train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(cifar10_mean, cifar10_std),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    
    test_transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    
    if normalize_test:
        test_transform_list.append(
            v2.Normalize(cifar10_mean, cifar10_std)
        )
    
    test_transform = v2.Compose(test_transform_list)
    
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    return train_loader, test_loader


def train_step(model, criterion, optimizer, batch):
    """Perform a single training step."""
    images, labels = batch
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    preds = model(images)
    loss = criterion(preds, labels)

    correct = (preds.argmax(dim=1) == labels).sum().item()
    total = labels.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), correct, total


def train(model, criterion, optimizer, scheduler, train_loader, num_epochs: int, tracker=None):
    """Train the model for specified number of epochs.
    
    Args:
        model: The model to train
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        num_epochs: Number of epochs to train
        tracker: Optional UpdateRatioTracker instance
    """
    model.train()  # Ensure model is in training mode
    import time

    losses = []
    steps = []
    step_count = 0

    train_start_time = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        correct = 0
        total = 0
        for step, batch in enumerate(train_loader, start=1):
            loss, _correct, _total = train_step(model, criterion, optimizer, batch)
            step_count += 1
            losses.append(loss)
            steps.append(step_count)
            correct += _correct
            total += _total
            if tracker is not None:
                tracker.record(model, optimizer)

        scheduler.step()
        duration = time.time() - start_time
        accuracy = 100 * correct / total
        if epoch % 2 == 0 or num_epochs == epoch - 1:
            print(f"Epoch {epoch} loss: {losses[-1]:.4f}, duration: {duration:.2f}s, accuracy: {accuracy:.2f}%")

    train_duration = time.time() - train_start_time
    print(f"Training duration: {train_duration:.2f}s")
    return losses, steps


@torch.no_grad()
def evaluate(model, test_loader, per_class_accuracy: bool = False):
    """Evaluate the model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        per_class_accuracy: Whether to print per-class accuracy (default: False)
    """
    model.eval()
    correct = 0
    total = 0
    count_correct_per_class = [0] * 10
    
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if per_class_accuracy:
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    count_correct_per_class[labels[i]] += 1

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f} %")
    
    if per_class_accuracy:
        for i in range(10):
            print(f"Class wise accuracy for class {i}: {100 * count_correct_per_class[i] / (total / 10):.2f} %")
    
    return accuracy


def check_model_outputs(model, train_loader, criterion):
    """Check initial model outputs to diagnose low loss issue"""
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, labels)

        # Get probabilities
        probs = torch.softmax(preds, dim=1)
        max_probs, predicted = torch.max(probs, dim=1)

        print(f"Initial loss: {loss.item():.4f}")
        print(f"Average max probability: {max_probs.mean().item():.4f}")
        print(f"Predicted classes (first 10): {predicted[:10].cpu().tolist()}")
        print(f"True classes (first 10): {labels[:10].cpu().tolist()}")
        print(f"Logits range: [{preds.min().item():.2f}, {preds.max().item():.2f}]")
        print(f"Logits mean: {preds.mean().item():.2f}, std: {preds.std().item():.2f}")

    model.train()


def plot_loss(losses, steps, window_size=100):
    """Plot training loss with optional moving average"""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))

    # Plot raw losses (can be noisy)
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, alpha=0.3, color='blue', label='Raw loss')

    # Calculate moving average if we have enough points
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        moving_avg_steps = steps[window_size-1:]
        plt.plot(moving_avg_steps, moving_avg, color='red', linewidth=2, label=f'Moving avg ({window_size} steps)')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (All Steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot smoothed version (every 100 steps for cleaner view)
    plt.subplot(1, 2, 2)
    if len(losses) >= 100:
        # Sample every 100th step for cleaner visualization
        sampled_steps = steps[::100]
        sampled_losses = losses[::100]
        plt.plot(sampled_steps, sampled_losses, marker='o', markersize=4, linewidth=2, color='green', label='Loss (every 100 steps)')
    else:
        plt.plot(steps, losses, marker='o', markersize=4, linewidth=2, color='green', label='Loss')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Sampled)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nLoss Statistics:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Min loss: {min(losses):.4f}")
    print(f"  Max loss: {max(losses):.4f}")
    print(f"  Average loss: {np.mean(losses):.4f}")

