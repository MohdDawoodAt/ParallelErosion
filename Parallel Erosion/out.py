import cv2
import numpy as np

# Read the eroded result from the text file
with open("eroded_result512omp.txt", "r") as file:
    lines = file.readlines()

# Convert the content to a NumPy array
result_array = np.array(
    [[int(pixel) for pixel in line.split()] for line in lines], dtype=np.uint8
)

# Create an OpenCV binary image from the NumPy array
result_image = result_array * 255  # Scale the values to 0 and 255

# Save the image as JPEG
cv2.imwrite("result20cuda.jpg", result_image)

print("Binary image saved as eroded_result.jpg")

# Read the eroded result from the text file
with open("eroded_result200cu.txt", "r") as file:
    lines = file.readlines()

# Convert the content to a NumPy array
result_array = np.array(
    [[int(pixel) for pixel in line.split()] for line in lines], dtype=np.uint8
)

# Create an OpenCV binary image from the NumPy array
result_image = result_array * 255  # Scale the values to 0 and 255

# Save the image as JPEG
cv2.imwrite("result2cuda.jpg", result_image)

print("Binary image saved as eroded_result.jpg")
# Read the eroded result from the text file
with open("eroded_result200omp.txt", "r") as file:
    lines = file.readlines()

# Convert the content to a NumPy array
result_array = np.array(
    [[int(pixel) for pixel in line.split()] for line in lines], dtype=np.uint8
)

# Create an OpenCV binary image from the NumPy array
result_image = result_array * 255  # Scale the values to 0 and 255

# Save the image as JPEG
cv2.imwrite("result200omp.jpg", result_image)

print("Binary image saved as eroded_result.jpg")

# Read the eroded result from the text file
with open("eroded_result512omp.txt", "r") as file:
    lines = file.readlines()

# Convert the content to a NumPy array
result_array = np.array(
    [[int(pixel) for pixel in line.split()] for line in lines], dtype=np.uint8
)

# Create an OpenCV binary image from the NumPy array
result_image = result_array * 255  # Scale the values to 0 and 255

# Save the image as JPEG
cv2.imwrite("result512omp.jpg", result_image)

print("Binary image saved as eroded_result.jpg")
