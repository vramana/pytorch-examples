## FF Net / ConvNet examples

Train the MNIST feedforward or convolutional network that lives in `ff_net/main.py`.

```bash
python ff_net/main.py --epochs 5
```

### Profiling the ConvNet

Use PyTorch's profiler to capture the ConvNet hotspots before porting layers to Triton:

```bash
python ff_net/main.py --profile --profile-steps 80 --trace-dir profiling/convnet
```

Then inspect the results with TensorBoard:

```bash
tensorboard --logdir profiling/convnet
```

Flags of interest:

- `--use-ffn`: profile or train the simple feedforward model instead of the ConvNet.
- `--batch-size`: override MNIST loader batch size.
- `--lr`: adjust the optimizer learning rate.

Setup:

```
git clone https://github.com/vramana/pytorch-examples.git
cd pytorch-examples/
uv sync
uv run --with jupyter jupyter lab --allow-root --no-browser --NotebookApp.token=''

```

**Stanford Courses**:

- [Stanford CS231N Deep Learning for Computer Vision 2025](https://www.youtube.com/watch?v=25zD5qJHYsk&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=4)
