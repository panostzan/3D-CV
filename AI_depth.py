
#
# GOAL:
# 1. Capture an image from our webcam.
# 2. Feed the image into a pretrained neural network.
# 3. Get a predicted depth value for each pixel.
# 4. Visualize the depth values as a thermal-style heat map.
#
# IMPORTANT:
# The model predicts RELATIVE depth, not actual metres.
# Higher predicted values generally mean closer surfaces.
# ============================================================


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Allows us to create our own red-to-purple colour map
from matplotlib.colors import LinearSegmentedColormap

# PIL: Converts images into a format accepted by our model
from PIL import Image

# Hugging Face: Loads and runs our pretrained neural network
from transformers import pipeline


# ============================================================
# STEP 1: CAPTURE AN IMAGE
# ============================================================
print("[1/6] Opening webcam...", flush=True)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press SPACE to capture an image, or Q to quit.")

while True:

    # Read one frame from the webcam
    # success = whether a frame was captured successfully
    # frame = image stored as a NumPy array (height, width, 3)
    success, frame = camera.read()

    if not success:
        raise RuntimeError("Could not read webcam.")

    # Display the current camera frame
    cv2.imshow("Webcam - Press SPACE", frame)

    # Read keyboard input
    key = cv2.waitKey(1) & 0xFF

    # ASCII code 32 represents SPACE
    if key == 32:
        print("[2/6] Image captured!", flush=True)
        break

    # Quit if the user presses Q
    elif key == ord("q"):
        camera.release()
        cv2.destroyAllWindows()
        raise SystemExit

# Release the webcam so other applications can use it
camera.release()

# Close the OpenCV camera window
cv2.destroyAllWindows()


# ============================================================
# STEP 2: PREPARE THE IMAGE FOR THE NEURAL NETWORK
# ============================================================

# OpenCV stores colours in BGR order (Blue, Green, Red).
# Our model expects a normal RGB image (Red, Green, Blue).
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert the NumPy array into a PIL image.
# The Hugging Face pipeline can accept this image format.
image = Image.fromarray(rgb)
print(f"[3/6] Image prepared! Size: {image.size}", flush=True)


# ============================================================
# STEP 3: LOAD THE PRETRAINED NEURAL NETWORK
# ============================================================

print("Loading Depth Anything V2...")

# Load the Small version of Depth Anything V2.
# The model has already been trained on image/depth data.
# We are NOT training it ourselves, only using it for inference.
#
# On the first run, the model weights download from the internet.
# On later runs, the downloaded files can be reused.
model = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
print("[4/6] Neural network loaded!", flush=True)

# ============================================================
# STEP 4: PREDICT DEPTH
# ============================================================

print("Predicting depth...")

# Pass our RGB image through the neural network.
# The network analyzes visual features and predicts depth.
import time

print("[5/6] Running depth prediction...", flush=True)

start = time.perf_counter()

result = model(image)

elapsed = time.perf_counter() - start

print(f"      Prediction completed in {elapsed:.2f} seconds!")

# Extract the predicted depth tensor.
#
# squeeze() removes unnecessary dimensions.
# cpu() moves the tensor into CPU memory if needed.
# numpy() converts the PyTorch tensor into a NumPy array.
#
# Final shape: (height, width)
depth = result["predicted_depth"].squeeze().cpu().numpy()

# Print information so we understand our data
print("RGB image shape:", rgb.shape)
print("Depth array shape:", depth.shape)

# Check the smallest and largest predicted values
print("Minimum predicted depth:", depth.min())
print("Maximum predicted depth:", depth.max())

# Inspect the predicted value at the centre of the depth map
h, w = depth.shape
print("Centre pixel prediction:", depth[h // 2, w // 2])


# ============================================================
# STEP 5: NORMALIZE DEPTH VALUES
# ============================================================

# Neural network outputs are not necessarily between 0 and 1.
#
# For visualization, we convert them to the range [0, 1].
#
# Smallest prediction becomes 0.
# Largest prediction becomes 1.
#
# These are relative values, NOT distances in metres.
#
# The tiny 1e-8 prevents division by zero if all values match.
depth_normalized = (depth - depth.min()) / (
    depth.max() - depth.min() + 1e-8
)


# ============================================================
# STEP 6: CREATE A THERMAL-STYLE COLOUR MAP
# ============================================================

# Matplotlib uses the order below from LOW to HIGH values.
#
# Low predicted value  -> Purple/Blue (farther)
# High predicted value -> Orange/Red (closer)
#
# The colours are only a visualization.
# They do not change the underlying depth predictions.
thermal = LinearSegmentedColormap.from_list(
    "thermal",
    ["purple", "blue", "yellow", "orange", "red"]
)


# ============================================================
# STEP 7: VISUALIZE THE RESULTS
# ============================================================

# Create a figure containing two plots, side by side.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LEFT: Show the original RGB image
axes[0].imshow(rgb)
axes[0].set_title("Original Webcam Image")

# RIGHT: Show the predicted depth using our colour map
heatmap = axes[1].imshow(
    depth_normalized,
    cmap=thermal,
    vmin=0,
    vmax=1
)

axes[1].set_title("AI Predicted Depth")

# Add a scale showing how colours correspond to values
fig.colorbar(
    heatmap,
    ax=axes[1],
    label="Relative proximity (higher = closer)"
)

# Hide the image axes to make the result cleaner
for ax in axes:
    ax.axis("off")

# Adjust spacing so the plots do not overlap
plt.tight_layout()

# Open the visualization window
print("[6/6] Depth heat map ready! Opening visualization...", flush=True)
plt.show()
