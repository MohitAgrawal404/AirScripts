import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cv2
import math

def load_point_cloud(pcd_file):
    # Load the point cloud from a PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    return pcd

def find_flat_areas(points, area_size=2.0, height_tolerance=0.1):
    # Create a KD-tree for efficient spatial querying (using X and Y for flat area detection)
    kdtree = cKDTree(points[:, [0, 1]])  # Only consider X and Y (forward/backward, left/right)

    flat_areas = []

    # Iterate over each point in the cloud
    for i, point in enumerate(points):
        # Query points within the given area size in X and Y (forward/back, left/right)
        neighbors_idx = kdtree.query_ball_point(point[0:2], area_size / 2.0)  # Radius = half of area_size
        neighbors = points[neighbors_idx]
        
        # Extract the neighbors' Z-values (heights)
        z_values = neighbors[:, 2]
        
        # Check if the variation in Z-values (heights) is below the tolerance (flat area)
        if np.max(z_values) - np.min(z_values) < height_tolerance:
            flat_areas.append(point)
    
    return np.array(flat_areas)

def visualize_point_cloud_with_flat_areas(points, flat_areas):
    # Create Open3D point cloud object
    colors = np.zeros((points.shape[0], 3))
    colors[:, 2] = 1  # Set all points to blue (0, 0, 1)
    
    # Set flat areas to green (0, 1, 0)
    for flat_point in flat_areas:
        distances = np.linalg.norm(points - flat_point, axis=1)
        closest_point_idx = np.argmin(distances)
        colors[closest_point_idx] = [0, 1, 0]  # Green for flat points

    # Create Open3D point cloud object with updated colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # Save the updated point cloud with colors to a new PCD file
    o3d.io.write_point_cloud("flat_areas_point_cloud_data.pcd", pcd)

def rotate_point_cloud(pcd, angle_deg=-90):
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Create the 2D rotation matrix for a clockwise rotation around the Z-axis (counterclockwise -90 degrees)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])

    # Apply the rotation matrix to each point in the point cloud
    points = np.asarray(pcd.points)
    rotated_points = np.dot(points, rotation_matrix.T)  # Dot product with the rotation matrix

    # Set the new points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(rotated_points)

    return pcd 

def apply_segmentation_to_point_cloud(pcd, segmentation_image, fx, fy, cx, cy):
    # Rotate the point cloud by -90 degrees before applying segmentation
    pcd = rotate_point_cloud(pcd, angle_deg=-90)
    
    points = np.asarray(pcd.points)

    # Initialize colors for the point cloud
    colors = np.zeros((points.shape[0], 3))

    # Loop through each point in the point cloud
    for i, (x, y, z) in enumerate(points):
        # Check if z is valid (non-zero and non-NaN)
        if z != 0 and not np.isnan(z):
            try:
                # Calculate the 2D image coordinates (u, v) corresponding to the 3D point (x, y, z)
                u = int((fx * x / z) + cx)  # Projection onto the image's horizontal axis
                v = int((fx * y / z) + cy)  # Projection onto the image's vertical axis

                # Check if the (u, v) coordinates fall within the image bounds
                if 0 <= u < segmentation_image.shape[1] and 0 <= v < segmentation_image.shape[0]:
                    # Get the segmentation color at (v, u)
                    color = segmentation_image[v, u, ::-1]  # Convert BGR to RGB
  # Assuming RGB format
                    colors[i] = color / 255.0  # Normalize to [0, 1] for Open3D

            except Exception as e:
                print(f"Error processing point {i}: {e}")
                continue
        else:
            print(f"Skipping point {i} due to invalid z value.")

    # Assign the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save or visualize the updated point cloud
    o3d.visualization.draw_geometries([pcd])




def calculate_focal_length_from_fov(image_width_px, image_height_px, fov_deg):
    """
    Calculate the focal lengths (fx, fy) in pixels using the FOV and image resolution.
    """
    fov_rad = math.radians(fov_deg)
    fx = (image_width_px / 2) / math.tan(fov_rad / 2)
    fy = (image_height_px / 2) / math.tan(fov_rad / 2)
    return fx, fy

def main():
    # Load the point cloud from the PCD file
    pcd_file = "point_cloud_data.pcd"
    pcd = load_point_cloud(pcd_file)
    
    # Get the points from the PCD
    points = np.asarray(pcd.points)

    # Adjust the area size and tolerance as needed
    area_size = 3  # Can be any value (e.g., 3x3 meters, 4x4 meters, etc.)
    height_tolerance = 0.1  # Define how flat the area needs to be

    # Find the flat areas based on the given tolerance for height (Z) and area size in X and Y
    flat_areas = find_flat_areas(points, area_size=area_size, height_tolerance=height_tolerance)

    # Visualize the point cloud with flat areas highlighted
    visualize_point_cloud_with_flat_areas(points, flat_areas)

    # Calculate focal lengths
    fx, fy = calculate_focal_length_from_fov(1080, 1920, 120)

    # Optionally, apply segmentation colors to the point cloud
    segmentation_image = cv2.imread("image_wide.png")  # Load the segmentation image
    cx, cy = 540, 960  # Camera intrinsics (example values)
    apply_segmentation_to_point_cloud(pcd, segmentation_image, fx, fy, cx, cy)

if __name__ == "__main__":
    main()
