import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('car_data.csv')


print("Dataset Head:\n", df.head())

df = df[['name', 'year', 'km_driven', 'fuel', 'transmission', 'owner', 'selling_price']]


le_fuel = LabelEncoder()
le_transmission = LabelEncoder()
le_owner = LabelEncoder()

df['fuel'] = le_fuel.fit_transform(df['fuel'])
df['transmission'] = le_transmission.fit_transform(df['transmission'])
df['owner'] = le_owner.fit_transform(df['owner'])


X = df[['year', 'km_driven', 'fuel', 'transmission', 'owner']]
y = df['selling_price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Performance:")
print("R² Score:", round(r2_score(y_test, y_pred), 2))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# Predict custom input
def predict_price():
    print("\nEnter Car Details to Predict Price:")
    year = int(input("Year of manufacture: "))
    km_driven = int(input("Kilometers driven: "))
    fuel = input("Fuel Type (Petrol/Diesel/CNG/LPG/Electric): ")
    transmission = input("Transmission (Manual/Automatic): ")
    owner = input("Owner Type (First Owner, Second Owner, etc.): ")

    # Encode inputs
    fuel_encoded = le_fuel.transform([fuel])[0]
    transmission_encoded = le_transmission.transform([transmission])[0]
    owner_encoded = le_owner.transform([owner])[0]

    features = [[year, km_driven, fuel_encoded, transmission_encoded, owner_encoded]]
    prediction = model.predict(features)[0]
    print(f"\nEstimated Selling Price: ₹ {round(prediction)}")

predict_price()
