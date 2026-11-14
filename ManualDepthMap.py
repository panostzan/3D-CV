
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# STEP 1: Create an artificial image (RGB)
height = 12
width = 18

# The background wall is light grey
image = np.zeros((height, width, 3), dtype=np.uint8)
# note image shape is 12,18,3 beccause each pixel has 3 channels 
# but depth shape is 12,18 only because each pixel contains just one depth value 
image[:] = [210, 215, 210]

# Draw a brown table
image[7:9, 2:16] = [150, 95, 55]
image[9:12, 3:5] = [110, 70, 45]
image[9:12, 13:15] = [110, 70, 45]

# Draw a red object in front of the wall
image[4:7, 7:11] = [210, 70, 55]


# STEP 2: Create the corresponding depth array
# Values represent distance in metres

depth = np.full((height, width), 4.0)

# The table is 2 metres away
depth[7:9, 2:16] = 2.0
depth[9:12, 3:5] = 2.0
depth[9:12, 13:15] = 2.0

# The red object is closer: 1 metre away
depth[4:7, 7:11] = 1.0


# STEP 3: Create a thermal-style colour map
# Low depth = red (close)
# High depth = blue/purple (far)

thermal = LinearSegmentedColormap.from_list(
    "thermal_depth",
    ["red", "orange", "yellow", "blue", "purple"]
)


# STEP 4: Display all three representations

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title("1. Original RGB Image")

# Depth array as numerical values
axes[1].imshow(depth, cmap="gray")
for row in range(height):
    for col in range(width):
        axes[1].text(
            col, row, f"{depth[row, col]:.0f}",
            ha="center", va="center", fontsize=6,
            color="red"
        )
axes[1].set_title("2. Depth Values (metres)")

# Thermal-style depth map
heatmap = axes[2].imshow(
    depth, cmap=thermal, vmin=1.0, vmax=4.0
)
axes[2].set_title("3. Depth Heat Map")

fig.colorbar(heatmap, ax=axes[2], label="Distance (m)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
