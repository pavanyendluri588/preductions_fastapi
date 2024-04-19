
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
# The entry point function MUST have two input arguments.
# If the input port is not connected, the corresponding
# dataframe argument will be None.
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

# Load the model and scalers from their respective files




def encode_df(df):
    # Use one-hot encoding for categorical columns
    if df.loc[0,"Sender_Country"] == "CANADA":
        df.loc[0,"Sender_Country_CANADA"] = 1
        df.loc[0,"Sender_Country_GERMANY"] = 0
        df.loc[0,"Sender_Country_OTHERS"] = 0
        df.loc[0,"Sender_Country_USA"] = 0

    elif df.loc[0,"Sender_Country"] == "GERMANY":
        df.loc[0,"Sender_Country_CANADA"] = 0
        df.loc[0,"Sender_Country_GERMANY"] = 1
        df.loc[0,"Sender_Country_OTHERS"] = 0
        df.loc[0,"Sender_Country_USA"] = 0
    elif df.loc[0,"Sender_Country"] == "OTHERS":
        df.loc[0,"Sender_Country_CANADA"] = 0
        df.loc[0,"Sender_Country_GERMANY"] = 0
        df.loc[0,"Sender_Country_OTHERS"] = 1
        df.loc[0,"Sender_Country_USA"] = 0
    elif df.loc[0,"Sender_Country"] == "USA":
        df.loc[0,"Sender_Country_CANADA"] = 0
        df.loc[0,"Sender_Country_GERMANY"] = 0
        df.loc[0,"Sender_Country_OTHERS"] = 0
        df.loc[0,"Sender_Country_USA"] = 1

    if df.loc[0,"Bene_Country"] == "CANADA":
        df.loc[0,"Bene_Country_CANADA"] = 1
        df.loc[0,"Bene_Country_GERMANY"] = 0
        df.loc[0,"Bene_Country_OTHERS"] = 0
        df.loc[0,"Bene_Country_USA"] = 0

    elif df.loc[0,"Bene_Country"] == "GERMANY":
        df.loc[0,"Bene_Country_CANADA"] = 0
        df.loc[0,"Bene_Country_GERMANY"] = 1
        df.loc[0,"Bene_Country_OTHERS"] = 0
        df.loc[0,"Bene_Country_USA"] = 0
    elif df.loc[0,"Bene_Country"] == "OTHERS":
        df.loc[0,"Bene_Country_CANADA"] = 0
        df.loc[0,"Bene_Country_GERMANY"] = 0
        df.loc[0,"Bene_Country_OTHERS"] = 1
        df.loc[0,"Bene_Country_USA"] = 0
    elif df.loc[0,"Bene_Country"] == "USA":
        df.loc[0,"Bene_Country_CANADA"] = 0
        df.loc[0,"Bene_Country_GERMANY"] = 0
        df.loc[0,"Bene_Country_OTHERS"] = 0
        df.loc[0,"Bene_Country_USA"] = 1

        
    if df.loc[0,"Transaction_Type"] == "MAKE-PAYMENT":
        df.loc[0,"Transaction_Type_MAKE-PAYMENT"] = 1
        df.loc[0,"Transaction_Type_MOVE-FUNDS"] = 0
        df.loc[0,"Transaction_Type_PAY-CHECK"] = 0
        df.loc[0,"Transaction_Type_QUICK-PAYMENT"] = 0
    elif df.loc[0,"Transaction_Type"] == "MOVE-FUNDS":
        df.loc[0,"Transaction_Type_MAKE-PAYMENT"] = 0
        df.loc[0,"Transaction_Type_MOVE-FUNDS"] = 1
        df.loc[0,"Transaction_Type_PAY-CHECK"] = 0
        df.loc[0,"Transaction_Type_QUICK-PAYMENT"] = 0
    elif df.loc[0,"Transaction_Type"] == "PAY-CHECK":
        df.loc[0,"Transaction_Type_MAKE-PAYMENT"] = 0
        df.loc[0,"Transaction_Type_MOVE-FUNDS"] = 0
        df.loc[0,"Transaction_Type_PAY-CHECK"] = 1
        df.loc[0,"Transaction_Type_QUICK-PAYMENT"] = 0
    elif df.loc[0,"Transaction_Type"] == "QUICK-PAYMENT":
        df.loc[0,"Transaction_Type_MAKE-PAYMENT"] = 0
        df.loc[0,"Transaction_Type_MOVE-FUNDS"] = 0
        df.loc[0,"Transaction_Type_PAY-CHECK"] = 0
        df.loc[0,"Transaction_Type_QUICK-PAYMENT"] = 1

    if df.loc[0,"Sender_Type"] == "BILL-COMPANY":
        df.loc[0,"Sender_Type_BILL-COMPANY"] = 1
        df.loc[0,"Sender_Type_CLIENT"] = 0
        df.loc[0,"Sender_Type_COMPANY"] = 0
        df.loc[0,"Sender_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Sender_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Sender_Type"] == "CLIENT":
        df.loc[0,"Sender_Type_BILL-COMPANY"] = 0
        df.loc[0,"Sender_Type_CLIENT"] = 1
        df.loc[0,"Sender_Type_COMPANY"] = 0
        df.loc[0,"Sender_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Sender_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Sender_Type"] == "COMPANY":
        df.loc[0,"Sender_Type_BILL-COMPANY"] = 0
        df.loc[0,"Sender_Type_CLIENT"] = 0
        df.loc[0,"Sender_Type_COMPANY"] = 1
        df.loc[0,"Sender_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Sender_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Sender_Type"] == "JPMC-CLIENT":
        df.loc[0,"Sender_Type_BILL-COMPANY"] = 0
        df.loc[0,"Sender_Type_CLIENT"] = 0
        df.loc[0,"Sender_Type_COMPANY"] = 0
        df.loc[0,"Sender_Type_JPMC-CLIENT"] = 1
        df.loc[0,"Sender_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Sender_Type"] == "JPMC-COMPANY":
        df.loc[0,"Sender_Type_BILL-COMPANY"] = 0
        df.loc[0,"Sender_Type_CLIENT"] = 0
        df.loc[0,"Sender_Type_COMPANY"] = 0
        df.loc[0,"Sender_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Sender_Type_JPMC-COMPANY"] = 1

    if df.loc[0,"Bene_Type"] == "BILL-COMPANY":
        df.loc[0,"Bene_Type_BILL-COMPANY"] = 1
        df.loc[0,"Bene_Type_CLIENT"] = 0
        df.loc[0,"Bene_Type_COMPANY"] = 0
        df.loc[0,"Bene_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Bene_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Bene_Type"] == "CLIENT":
        df.loc[0,"Bene_Type_BILL-COMPANY"] = 0
        df.loc[0,"Bene_Type_CLIENT"] = 1
        df.loc[0,"Bene_Type_COMPANY"] = 0
        df.loc[0,"Bene_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Bene_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Bene_Type"] == "COMPANY":
        df.loc[0,"Bene_Type_BILL-COMPANY"] = 0
        df.loc[0,"Bene_Type_CLIENT"] = 0
        df.loc[0,"Bene_Type_COMPANY"] = 1
        df.loc[0,"Bene_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Bene_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Bene_Type"] == "JPMC-CLIENT":
        df.loc[0,"Bene_Type_BILL-COMPANY"] = 0
        df.loc[0,"Bene_Type_CLIENT"] = 0
        df.loc[0,"Bene_Type_COMPANY"] = 0
        df.loc[0,"Bene_Type_JPMC-CLIENT"] = 1
        df.loc[0,"Bene_Type_JPMC-COMPANY"] = 0
    elif df.loc[0,"Bene_Type"] == "JPMC-COMPANY":
        df.loc[0,"Bene_Type_BILL-COMPANY"] = 0
        df.loc[0,"Bene_Type_CLIENT"] = 0
        df.loc[0,"Bene_Type_COMPANY"] = 0
        df.loc[0,"Bene_Type_JPMC-CLIENT"] = 0
        df.loc[0,"Bene_Type_JPMC-COMPANY"] = 1
    df.drop(['Sender_Country', 'Bene_Country', 'Transaction_Type', 'Sender_Type', 'Bene_Type'],axis=1,inplace=True)

    #df = pd.get_dummies(df, columns=['Sender_Country', 'Bene_Country', 'Transaction_Type', 'Sender_Type', 'Bene_Type'], dtype=int)
    print("encode_df:================================================================\n",df.columns)
    return df
def transform_df(df):
    # Define a function to categorize countries
    def categorize_country(country):
        if country == 'USA':
            return 'USA'
        elif country == 'CANADA':
            return 'CANADA'
        elif country == 'GERMANY':
            return 'GERMANY'
        else:
            return 'OTHERS'

    # Drop rows with null values
    df.dropna(inplace=True)

    # Apply country categorization to 'Sender_Country' and 'Bene_Country' columns
    df['Sender_Country'] = df['Sender_Country'].apply(categorize_country)
    df['Bene_Country'] = df['Bene_Country'].apply(categorize_country)

    # Extract sender type and bene type from respective IDs
    df["Sender_Type"] = df["Sender_Id"].apply(lambda sender_id: "-".join(sender_id.split("-")[:-1]) if "-" in sender_id else sender_id)
    df["Bene_Type"] = df["Bene_Id"].apply(lambda sender_id: "-".join(sender_id.split("-")[:-1]) if "-" in sender_id else sender_id)

    # Split 'Time_step' into 'Date' and 'Time', then convert 'Time' to seconds
    df['Date'] = df['Time_step'].str.split(" ").str[0]
    df['Time'] = df['Time_step'].str.split(" ").str[1]
    df['Time'] = df['Time'].apply(lambda x: int(x.split(":")[0]) * 3600 + int(x.split(":")[1]) * 60 + int(x.split(":")[2]))

    # Extract 'Year', 'Month', and 'Day' from 'Date'
    df[['Year', 'Month', 'Day']] = df['Date'].str.split('-', expand=True)

    # Drop unnecessary columns
    df.drop(['Transaction_Id','Time_step','Sender_Id','Sender_Account','Sender_lob','Bene_Id','Bene_Account','Date'], axis=1, inplace=True)

    return df


def scaling_df(df):

    # Initialize StandardScaler
    #scaler_standard = StandardScaler()

    # Fit and transform the data
    scaler_standard = pickle.load(open('scaler_standard_Time.pkl', 'rb'))
    df['Time_Scaled_Standard'] = scaler_standard.transform(df[['Time']])

    # Initialize MinMaxScaler
    #scaler_minmax = MinMaxScaler()

    # Fit and transform the data
     
    scaler_minmax_Day = pickle.load(open('scaler_minmax_Day.pkl', 'rb')) 
    scaler_minmax_Month = pickle.load(open('scaler_minmax_Month.pkl', 'rb')) 
    scaler_minmax_Year = pickle.load(open('scaler_minmax_Year.pkl', 'rb'))
    df['Year_MinMax'] = scaler_minmax_Year.transform(df[['Year']])
    df['Month_MinMax'] = scaler_minmax_Month.transform(df[['Month']])
    df['Day_MinMax'] = scaler_minmax_Day.transform(df[['Day']])

    df.drop(['Time','Year', 'Month', 'Day'],axis=1,inplace=True)
    print("scaling_df:==========================\n",df.columns)
    return df

def balance_df(df):
    # Assuming your data is in a DataFrame called 'data'
    # Using SMOTE to oversample the minority class
    if df.shape[0] == 1063398:
        X = df.drop('Label', axis=1)
        y = df['Label']
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        # Creating a new balanced DataFrame
        df = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=['Label'])], axis=1)

    return df

def do_transformations(df):
    print(df.columns)
    df = encode_df(scaling_df(df.drop("Transaction_Id",axis=1)))
    print(df.columns)
    return df

if __name__ == '__main__':
    path = "transformed.csv"
    df = pd.read_csv(path)
    print(do_transformations(df))

