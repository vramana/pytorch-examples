import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


class FeedforwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def get_data(batch_size: int):
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_step(model, criterion, optimizer, batch):
    images, labels = batch
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    preds = model(images)
    loss = criterion(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, criterion, optimizer, train_loader, num_epochs: int):
    for epoch in range(num_epochs):
        print(f"starting epoch {epoch}")
        for step, batch in enumerate(train_loader, start=1):
            loss = train_step(model, criterion, optimizer, batch)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss:.4f}")
        print(f"ends epoch {epoch}")


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f} %")
    return accuracy


def available_activities():
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def profile_training(model, criterion, optimizer, train_loader, steps: int, trace_dir: str):
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(trace_dir, worker_name="convnet")
    activities = available_activities()

    data_iter = iter(train_loader)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=handler,
    ) as prof:
        for step in range(steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            train_step(model, criterion, optimizer, batch)
            prof.step()
    print(f"Profiler traces written to {trace_dir}. Launch TensorBoard with `tensorboard --logdir {trace_dir}`.")


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST ConvNet with optional Torch Profiler run")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--profile", action="store_true", help="Run a short profiler session instead of full training")
    parser.add_argument("--profile-steps", type=int, default=50, help="Number of training steps to profile")
    parser.add_argument("--trace-dir", type=str, default="profiling", help="Directory to store profiler traces")
    parser.add_argument("--use-ffn", action="store_true", help="Train the simple feedforward network instead of the ConvNet")
    return parser.parse_args()


def main():
    args = parse_args()
    num_classes = 10
    model = FeedforwardNet(28 * 28, 500, num_classes) if args.use_ffn else ConvNet(num_classes)
    model = model.to(DEVICE)

    train_loader, test_loader = get_data(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.profile:
        print(f"Running profiler for {args.profile_steps} steps...")
        profile_training(model, criterion, optimizer, train_loader, args.profile_steps, args.trace_dir)
    else:
        train(model, criterion, optimizer, train_loader, args.epochs)
        evaluate(model, test_loader)


if __name__ == "__main__":
    main()

