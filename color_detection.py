import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# Function to identify color names
def get_color_name(r, g, b):

    if r < 30 and g < 30 and b < 30:
        return "Black"

    elif r > 220 and g > 220 and b > 220:
        return "White"

    elif abs(r-g) < 20 and abs(g-b) < 20:
        if r > 150:
            return "Light Gray"
        else:
            return "Gray"

    elif b > r and b > g:
        if b < 100:
            return "Dark Blue"
        elif b < 180:
            return "Blue"
        else:
            return "Light Blue"

    elif g > r and g > b:
        if g < 100:
            return "Dark Green"
        else:
            return "Green"

    elif r > g and r > b:
        if g > 150:
            return "Orange"
        else:
            return "Red"

    elif r > 120 and b > 120:
        return "Purple"

    elif r > 180 and g > 180 and b < 100:
        return "Yellow"

    else:
        return "Unknown"


# Upload image
print("Upload an image")
uploaded = files.upload()

image_path = next(iter(uploaded))

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error loading image")

else:
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten image
    pixels = rgb_image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Number of dominant colors
    k = 5

    # K-Means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Apply K-Means
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    # Count pixels
    counts = np.bincount(labels.flatten())

    # Sort by frequency
    sorted_idx = np.argsort(-counts)

    print("\nTop 5 Dominant Colors\n")

    for i, idx in enumerate(sorted_idx):
        color = centers[idx]

        r = int(color[0])
        g = int(color[1])
        b = int(color[2])

        color_name = get_color_name(r, g, b)

        print(
            f"Color {i+1}: {color_name:<12} RGB({r}, {g}, {b})"
        )

    # Show original image
    print("\nOriginal Image:")
    cv2_imshow(image)

    # Create palette
    palette = np.zeros((100, 500, 3), dtype=np.uint8)

    start = 0
    total = np.sum(counts)

    for idx in sorted_idx:
        end = start + int((counts[idx] / total) * 500)
        palette[:, start:end] = centers[idx]
        start = end

    palette_bgr = cv2.cvtColor(
        palette,
        cv2.COLOR_RGB2BGR
    )

    print("\nDetected Color Palette:")
    cv2_imshow(palette_bgr)