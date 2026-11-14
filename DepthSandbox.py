
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


# ============================================================
# STEP 6: PREPARE OUR 3D POINT CLOUD
# ============================================================

# Our depth model produces RELATIVE proximity predictions:
# Higher normalized values = closer to the camera.
#
# For our 3D experiment, we'll turn them into illustrative
# distances between 1 and 4 arbitrary units.
#
# These are NOT real-world metres!
z_map = 1.0 + 3.0 * (1.0 - depth_normalized)

# Get the depth map's height and width
h, w = depth_normalized.shape

# Our first camera model uses an APPROXIMATE focal length.
# Proper camera calibration would give us a better value.
f = 0.9 * w

# Sample every 8th pixel to keep the 3D plot lightweight.
step = 8

# Create grids containing pixel coordinates (u, v).
# Every pixel has a horizontal and vertical position.
u, v = np.meshgrid(
    np.arange(0, w, step),
    np.arange(0, h, step)
)

# Get the estimated depth at each sampled pixel.
z = z_map[::step, ::step]

# Convert pixel coordinates into approximate 3D positions.
#
# x = horizontal position in 3D
# y = vertical position in 3D
# z = estimated distance from camera
#
# We flip y because image coordinates increase downward.
x = (u - w / 2) * z / f
y = -(v - h / 2) * z / f

# Use the actual RGB image to colour the 3D points.
# Resize if the model returned a different image resolution.
rgb_resized = cv2.resize(rgb, (w, h))

# Matplotlib expects RGB values between 0 and 1.
colors = rgb_resized[::step, ::step] / 255.0


# ============================================================
# STEP 7: CREATE THE THREE-PANEL SANDBOX
# ============================================================

# Create a figure with three plots.
fig = plt.figure(figsize=(16, 6))

# Panel 1: Original webcam image
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(rgb_resized)
ax1.set_title("1. Original Webcam Image")
ax1.axis("off")

# Panel 2: Predicted depth heat map
ax2 = fig.add_subplot(1, 3, 2)

# Use the same thermal colour scheme as our last project.
from matplotlib.colors import LinearSegmentedColormap

thermal = LinearSegmentedColormap.from_list(
    "thermal",
    ["purple", "blue", "yellow", "orange", "red"]
)

ax2.imshow(
    depth_normalized,
    cmap=thermal,
    vmin=0,
    vmax=1
)

ax2.set_title("2. Click to Inspect Depth")
ax2.axis("off")

# Add a text label showing our selected pixel's depth.
depth_label = ax2.text(
    0.5, -0.08,
    "Click anywhere on the depth map",
    transform=ax2.transAxes,
    ha="center",
    fontsize=10
)

# Add a marker that moves to the clicked position.
marker, = ax2.plot(
    [], [], "wo",
    markersize=8,
    markeredgecolor="black"
)


# Panel 3: Interactive 3D point cloud
ax3 = fig.add_subplot(1, 3, 3, projection="3d")

# Draw a 3D point for each sampled pixel.
ax3.scatter(
    x.flatten(),
    y.flatten(),
    z.flatten(),
    c=colors.reshape(-1, 3),
    s=1
)

ax3.set_title("3. Rotate the Point Cloud")

# Label the three spatial axes.
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z (relative)")

# Choose a starting viewing angle.
ax3.view_init(elev=20, azim=-70)

# Keep the plot's proportions reasonably balanced.
ax3.set_box_aspect((1, 1, 1))


# ============================================================
# STEP 8: MAKE THE DEPTH MAP CLICKABLE
# ============================================================

def on_click(event):
    # Only respond to clicks inside the depth-map panel.
    if event.inaxes != ax2:
        return

    # Convert mouse position into image pixel coordinates.
    col = int(round(event.xdata))
    row = int(round(event.ydata))

    # Make sure the selected pixel is inside the image.
    if not (0 <= row < h and 0 <= col < w):
        return

    # Define a square around our selected pixel.
    # radius=5 gives us up to an 11x11 region.
    radius = 5

    # Keep our square inside the image boundaries.
    y1 = max(0, row - radius)
    y2 = min(h, row + radius + 1)

    x1 = max(0, col - radius)
    x2 = min(w, col + radius + 1)

    # Extract the depth values in this small region.
    region = depth_normalized[y1:y2, x1:x2]

    # Calculate the average predicted proximity.
    average = np.mean(region)

    # Display the result in our plot.
    depth_label.set_text(
        f"Pixel ({col}, {row}) | Avg proximity: {average:.3f}"
    )

    # Move the marker to the selected position.
    marker.set_data([col], [row])

    # Print the result to the terminal too.
    print(
        f"Pixel ({col}, {row}) | "
        f"Average relative proximity: {average:.3f}"
    )

    # Refresh the figure to show the new marker and label.
    fig.canvas.draw_idle()


# Connect mouse clicks to our function.
fig.canvas.mpl_connect("button_press_event", on_click)


# ============================================================
# STEP 9: OPEN THE INTERACTIVE VIEWER
# ============================================================

print("Depth Explorer ready!")
print("Click the heat map to inspect depth.")
print("Drag the 3D plot to rotate the point cloud.")

plt.tight_layout()
plt.show()
