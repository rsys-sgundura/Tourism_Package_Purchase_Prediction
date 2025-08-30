# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'     # Target variable indicating whether the customer has purchased a package (0: No, 1: Yes).

# Drop the unique identifier
tourism_dataset.drop(columns=['CustomerID'], inplace=True)

# List of numerical features in the dataset
numeric_features = [
    'Age',                      # Age of the customer i.e. 41.0,29.0,47.0
    'CityTier',                 # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3). i.e. 1,2,3
    'DurationOfPitch',          # Duration of the sales pitch delivered to the customer. i.e. 6.0,4.0,8.0,9.0
    'NumberOfPersonVisiting',   # Total number of people accompanying the customer on the trip. i.e. 3,2,3
    'NumberOfFollowups',        # Total number of follow-ups by the salesperson after the sales pitch. i.e. 3.0,2.0
    'PreferredPropertyStar',    # Preferred hotel rating by the customer. i.e. 3.0,4.0,5.0
    'NumberOfTrips',            # Average number of trips the customer takes annually. i.e. 1.0,2.0,7.0,5.0
    'Passport',                 # Whether the customer holds a valid passport (0: No, 1: Yes).
    'OwnCar',                   # Whether the customer owns a car (0: No, 1: Yes).
    'PitchSatisfactionScore',   # Score indicating the customer's satisfaction with the sales pitch. i.e. 2,3
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer. i.e. 0.0,2.0,5.0
    'MonthlyIncome',            # Gross monthly income of the customer.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer was contacted (Company Invited or Self Inquiry).
    'Occupation',            # Customer's occupation (e.g., Salaried, Free lancer).
    'Gender',                # Gender of the customer (Male, Female, Fe male ).
    'ProductPitched',        # The type of product pitched to the customer.
    'MaritalStatus',         # Marital status of the customer (Single, Married, Divorced).
    'Designation',           # Customer's designation in their current organization.
]

tourism_dataset['Gender']=tourism_dataset['Gender'].replace({'Fe Male' : 'Female'})

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]  #18 columns in total here .

# Define target variable
y = tourism_dataset[target]


# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="SudeendraMG/tourism-package-purchase-prediction",
        repo_type="dataset",
    )
