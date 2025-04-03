import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load JSON data
with open('/Users/mangus/Desktop/squalla-disc-search-prod.discs3.json') as f:
    data = json.load(f)

# Extract relevant fields and clean data
discs = []
for disc in data:
    try:
        # Ensure all required fields are present and valid
        if all(key in disc and disc[key] not in [None, ''] for key in ['infinite_ratings', 'diameter', 'height', 'rim_depth', 'rim_width', 'rim_thickness', 'rim_diameter_ratio', 'rim_configuration']):
            # Convert and validate numeric fields
            ratings = list(map(float, disc['infinite_ratings'].split('/')))
            rim_diameter_ratio = disc['rim_diameter_ratio']
            if rim_diameter_ratio is not None:
                rim_diameter_ratio = float(rim_diameter_ratio.replace('%', '')) / 100  # Convert percentage to decimal
            else:
                rim_diameter_ratio = 0.0  # Default value if rim_diameter_ratio is missing or None

            # Validate and clean rim_configuration
            rim_configuration = disc['rim_configuration']
            try:
                rim_configuration = float(rim_configuration)  # Ensure it's a valid float
            except (ValueError, TypeError):
                continue  # Skip this disc if rim_configuration is invalid

            dimensions = {
                'diameter': float(disc['diameter']),
                'height': float(disc['height']),
                'rim_depth': float(disc['rim_depth']),
                'rim_width': float(disc['rim_width'].replace(' cm', '')),
                'rim_thickness': float(disc['rim_thickness']),
                'rim_diameter_ratio': rim_diameter_ratio,
                'rim_configuration': rim_configuration  # Include cleaned rim_configuration
            }
            # Append only if all ratings and dimensions are valid
            if len(ratings) == 4:
                discs.append({**dimensions, 'speed': ratings[0], 'glide': ratings[1], 'turn': ratings[2], 'fade': ratings[3]})
    except (KeyError, ValueError, TypeError):
        # Skip discs with invalid or missing data
        continue

# Create DataFrame
df = pd.DataFrame(discs)

# Features and target variables
X = df[['diameter', 'height', 'rim_depth', 'rim_width', 'rim_thickness', 'rim_diameter_ratio', 'rim_configuration']]
y = df[['speed', 'glide', 'turn', 'fade']]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Mean Squared Error for Speed, Glide, Turn, Fade: {mse}")

# Function to predict speed, glide, turn, and fade for a given disc
def predict_disc(dimensions):
    """
    Predict the speed, glide, turn, and fade for a disc based on its dimensions.

    Parameters:
        dimensions (dict): A dictionary containing 'diameter', 'height', 'rim_depth', 'rim_width', 'rim_thickness',
                           'rim_diameter_ratio', and 'rim_configuration'.

    Returns:
        dict: Predicted values for speed, glide, turn, and fade.
    """
    input_data = pd.DataFrame([dimensions])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    return {
        'speed': round(prediction[0], 1),
        'glide': round(prediction[1], 1),
        'turn': round(prediction[2], 1),
        'fade': round(prediction[3], 1)
    }

# Analyze every disc in the JSON and add predicted ratings
for disc in data:
    try:
        # Ensure all required fields are present and valid
        if all(key in disc for key in ['diameter', 'height', 'rim_depth', 'rim_width', 'rim_thickness', 'rim_diameter_ratio', 'rim_configuration']):
            rim_diameter_ratio = disc['rim_diameter_ratio']
            if rim_diameter_ratio is not None:
                rim_diameter_ratio = float(rim_diameter_ratio.replace('%', '')) / 100  # Convert percentage to decimal
            else:
                rim_diameter_ratio = 0.0  # Default value if rim_diameter_ratio is missing or None

            # Validate and clean rim_configuration
            rim_configuration = disc['rim_configuration']
            try:
                rim_configuration = float(rim_configuration)  # Ensure it's a valid float
            except (ValueError, TypeError):
                continue  # Skip this disc if rim_configuration is invalid

            dimensions = {
                'diameter': float(disc['diameter']),
                'height': float(disc['height']),
                'rim_depth': float(disc['rim_depth']),
                'rim_width': float(disc['rim_width'].replace(' cm', '')),
                'rim_thickness': float(disc['rim_thickness']),
                'rim_diameter_ratio': rim_diameter_ratio,
                'rim_configuration': rim_configuration  # Include cleaned rim_configuration
            }
            # Predict ratings
            predicted = predict_disc(dimensions)
            # Add squalla_ratings field in the format speed/glide/turn/fade
            disc['squalla_ratings'] = f"{predicted['speed']}/{predicted['glide']}/{predicted['turn']}/{predicted['fade']}"
    except (KeyError, ValueError):
        # Skip discs with invalid or missing data
        continue

# Save the updated data to a new JSON file
output_file = '/Users/mangus/Desktop/squalla-disc-search-prod-with-squalla-ratings.json'
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Updated JSON file with squalla_ratings saved to {output_file}")

# Calculate the average difference between infinite_ratings and squalla_ratings
differences = {'speed': [], 'glide': [], 'turn': [], 'fade': []}
valid_indices = []

for i, disc in enumerate(data):
    try:
        # Ensure infinite_ratings and squalla_ratings are present and valid
        if 'infinite_ratings' in disc and 'squalla_ratings' in disc:
            infinite_ratings = list(map(float, disc['infinite_ratings'].split('/')))
            squalla_ratings = list(map(float, disc['squalla_ratings'].split('/')))

            # Calculate differences for each value
            differences['speed'].append(abs(infinite_ratings[0] - squalla_ratings[0]))
            differences['glide'].append(abs(infinite_ratings[1] - squalla_ratings[1]))
            differences['turn'].append(abs(infinite_ratings[2] - squalla_ratings[2]))
            differences['fade'].append(abs(infinite_ratings[3] - squalla_ratings[3]))

            # Track the index of this valid disc
            valid_indices.append(i)
    except (KeyError, ValueError):
        # Skip discs with invalid or missing data
        continue

# Calculate the average difference for each value
average_differences = {key: sum(values) / len(values) if values else 0 for key, values in differences.items()}

# Find the single biggest and smallest differences for each value
biggest_differences = {key: max(values) if values else 0 for key, values in differences.items()}
smallest_differences = {key: min(values) if values else 0 for key, values in differences.items()}

# Calculate the number of differences greater than 1 for each value
differences_greater_than_one = {key: sum(1 for value in values if value > 1) for key, values in differences.items()}

# Calculate the number of differences under 1 for each value
differences_under_one = {key: sum(1 for value in values if value < 1) for key, values in differences.items()}

# Calculate the number of times all four ratings are greater than 1
all_greater_than_one = sum(
    1 for i in range(len(differences['speed']))
    if all(differences[key][i] > 1 for key in differences.keys())
)

# Calculate the number of times all four ratings are under 1
all_under_one = sum(
    1 for i in range(len(differences['speed']))
    if all(differences[key][i] < 1 for key in differences.keys())
)

# Find the titles of objects where all four ratings are greater than 1
titles_all_greater_than_one = [
    data[valid_indices[i]]['title'] for i in range(len(valid_indices))
    if 'title' in data[valid_indices[i]] and all(differences[key][i] > 1 for key in differences.keys())
]

# Print the average differences
print("Average Differences Between Infinite Ratings and Squalla Ratings:")
for key, value in average_differences.items():
    print(f"{key.capitalize()}: {value:.2f}")

# Print the biggest and smallest differences
print("\nBiggest and Smallest Differences Between Infinite Ratings and Squalla Ratings:")
for key in differences.keys():
    print(f"{key.capitalize()}:")
    print(f"  Biggest Difference: {biggest_differences[key]:.2f}")
    print(f"  Smallest Difference: {smallest_differences[key]:.2f}")

# Print the number of differences greater than 1
print("\nNumber of Differences Greater Than 1 Between Infinite Ratings and Squalla Ratings:")
for key, count in differences_greater_than_one.items():
    print(f"{key.capitalize()}: {count}")

# Print the number of differences under 1
print("\nNumber of Differences Under 1 Between Infinite Ratings and Squalla Ratings:")
for key, count in differences_under_one.items():
    print(f"{key.capitalize()}: {count}")

# Print the number of times all four ratings are greater than 1
print("\nNumber of Times All Four Ratings Are Greater Than 1:")
print(f"All Greater Than 1: {all_greater_than_one}")

# Print the number of times all four ratings are under 1
print("\nNumber of Times All Four Ratings Are Under 1:")
print(f"All Under 1: {all_under_one}")

# Print the titles of objects where all four ratings are greater than 1
print("\nTitles of Objects Where All Four Ratings Are Greater Than 1:")
for title in titles_all_greater_than_one:
    print(f"- {title}")
