def adjust_flight_ratings(speed, glide, turn, fade, throw_speed, plastic_type, release_angle):
    """
    Adjust the flight ratings of a disc based on the throw speed, plastic type, and release angle.

    Parameters:
        speed (float): The original speed rating of the disc.
        glide (float): The original glide rating of the disc.
        turn (float): The original turn rating of the disc.
        fade (float): The original fade rating of the disc.
        throw_speed (float): The speed of the throw in mph.
        plastic_type (str): The type of plastic the disc is made from.
        release_angle (float): The release angle of the disc in degrees (-45 to +45).

    Returns:
        dict: Adjusted flight ratings with turn and fade modified.
    """
    # Map of intended speeds for each speed rating
    speed_to_intended_speed = {
        1: 10, 2: 15, 3: 20,
        4: 25, 5: 30, 6: 35,
        7: 40, 8: 45, 9: 50,
        10: 55, 11: 60, 12: 65,
        13: 70, 14: 75
    }

    # Stability modifiers for different plastic types
    plastic_stability_modifiers = {
        'base': -0.5,
        'mid-grade': -0.2,
        'premium': 0.2,
        'flexible': -0.3,
        'overmold': 0.3,
        'lightweight': -0.4,
        'glow': 0.1
    }

    # Dynamically determine the intended speed based on the speed rating
    intended_speed = speed_to_intended_speed.get(int(round(speed)), 60)  # Default to 60 if speed is not in the map

    # Calculate the scaling factor for adjustments
    speed_factor = throw_speed / intended_speed

    # Get the stability modifier for the plastic type
    stability_modifier = plastic_stability_modifiers.get(plastic_type.lower(), 0)

    # Adjust turn and fade values based on throw speed and plastic type
    adjusted_turn = (turn * speed_factor) + stability_modifier
    adjusted_fade = (fade * (intended_speed / throw_speed)) + stability_modifier

    # Adjust turn and fade values based on release angle
    adjusted_turn += release_angle / 45  # Positive release angle increases turn
    adjusted_fade -= release_angle / 45  # Positive release angle decreases fade

    # Return the adjusted ratings with turn and fade modified
    return {
        'speed': round(speed, 1),  # Unchanged
        'glide': round(glide, 1),  # Unchanged
        'turn': round(adjusted_turn, 1),  # Adjusted
        'fade': round(adjusted_fade, 1)  # Adjusted
    }

# Example usage
original_ratings = {'speed': 12, 'glide': 5, 'turn': -2, 'fade': 3}
throw_speed = 65  # Throw speed in mph
plastic_type = 'glow'  # Plastic type of the disc
release_angle = -20  # Release angle in degrees (e.g., -20 for hyzer)

adjusted_ratings = adjust_flight_ratings(
    original_ratings['speed'],
    original_ratings['glide'],
    original_ratings['turn'],
    original_ratings['fade'],
    throw_speed,
    plastic_type,
    release_angle
)

print("Original Ratings:", original_ratings)
print("Adjusted Ratings for Throw Speed", throw_speed, "mph, Plastic Type", plastic_type, "and Release Angle", release_angle, "°:", adjusted_ratings)