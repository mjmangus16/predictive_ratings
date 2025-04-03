import cv2
import numpy as np
import bpy

def process_image(image_path):
    """
    Process the image to extract the disc's shape and dimensions.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the disc
    disc_contour = max(contours, key=cv2.contourArea)

    # Approximate the disc as a circle
    (x, y), radius = cv2.minEnclosingCircle(disc_contour)

    return x, y, radius

def create_3d_model(x, y, radius, output_path):
    """
    Create a 3D model of the disc using Blender.
    """
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Create a cylinder to represent the disc
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64,
        radius=radius / 100,  # Scale down for Blender units
        depth=0.02,  # Approximate disc thickness
        location=(0, 0, 0)
    )

    # Save the 3D model
    bpy.ops.export_scene.obj(filepath=output_path)

def main(image_path, output_path):
    """
    Main function to process the image and create a 3D model.
    """
    # Process the image
    x, y, radius = process_image(image_path)

    # Create the 3D model
    create_3d_model(x, y, radius, output_path)

    print(f"3D model saved to {output_path}")

# Example usage
image_path = "disc_image.jpg"  # Path to the input image
output_path = "disc_model.obj"  # Path to save the 3D model
main(image_path, output_path)