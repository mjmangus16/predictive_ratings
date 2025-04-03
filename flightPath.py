import matplotlib.pyplot as plt
import numpy as np

def simulate_flight_path(speed, glide, turn, fade):
    """
    Simulate the flight path of a disc based on its flight ratings.

    Parameters:
        speed (float): The speed rating of the disc.
        glide (float): The glide rating of the disc.
        turn (float): The turn rating of the disc.
        fade (float): The fade rating of the disc.

    Returns:
        list: A list of (x, y) points representing the flight path.
    """
    # Initialize the flight path
    flight_path = []

    # Simulate the flight
    x = 0
    y = 0
    velocity = speed * 10  # Scale speed to velocity
    angle = 0  # Initial angle of the disc
    for t in np.linspace(0, 5, 100):  # Simulate for 5 seconds with 100 points
        # Update position based on velocity and angle
        x += velocity * np.cos(np.radians(angle)) * 0.1
        y += velocity * np.sin(np.radians(angle)) * 0.1

        # Adjust angle based on turn and fade
        if t < 2.5:  # First half of the flight
            angle += turn * 0.1
        else:  # Second half of the flight
            angle -= fade * 0.1

        # Reduce velocity based on glide
        velocity *= (1 - 0.01 * glide)

        # Stop if the disc hits the ground
        if y < 0:
            y = 0
            break

        # Append the current position to the flight path
        flight_path.append((x, y))

    return flight_path

# Example: Simulate and plot the flight path for a disc
squalla_flight_ratings = {'speed': 7, 'glide': 5, 'turn': -1, 'fade': 3}
flight_path = simulate_flight_path(
    squalla_flight_ratings['speed'],
    squalla_flight_ratings['glide'],
    squalla_flight_ratings['turn'],
    squalla_flight_ratings['fade']
)

# Debug: Print the flight path
print("Flight Path:", flight_path)

# Extract x and y points
x_points = [point[0] for point in flight_path]
y_points = [point[1] for point in flight_path]

# Debug: Print x and y points
print("X Points:", x_points)
print("Y Points:", y_points)

# Plot the flight path
plt.figure(figsize=(10, 5))
plt.plot(x_points, y_points, label='Flight Path')
plt.title('Disc Flight Path')
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Height')
plt.legend()
plt.grid()

# Adjust plot range if necessary
plt.xlim(0, max(x_points) + 10)
plt.ylim(0, max(y_points) + 10)

plt.show()