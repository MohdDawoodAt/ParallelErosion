import cv2
import numpy as np


def resize_image(input_path, output_path, width, height):
    # Read the image using OpenCV
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image
    resized_img = cv2.resize(img, (width, height))

    # Save the resized image
    cv2.imwrite(output_path, resized_img)


def image_to_binary_matrix(image_path):
    # Open the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a binary matrix
    binary_matrix = (img > 128).astype(np.int)

    return binary_matrix


def write_binary_matrix_to_file(binary_matrix, output_file):
    with open(output_file, "w") as f:
        for row in binary_matrix:
            f.write(" ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    # Specify the input image path
    input_image_path = "erosion.png"

    # Resize the image to 512x512
    resize_image(input_image_path, "200.jpg", 200, 200)

    # Resize the image to 1000x1000
    resize_image(input_image_path, "512.jpg", 512, 512)

    # Convert resized images to binary matrices
    binary_matrix_512x512 = image_to_binary_matrix("200.jpg")
    binary_matrix_1000x1000 = image_to_binary_matrix("512.jpg")

    # Write binary matrices to text files
    write_binary_matrix_to_file(binary_matrix_512x512, "200.txt")
    write_binary_matrix_to_file(binary_matrix_1000x1000, "512.txt")
