import pandas as pd
import json

categorical_featues = [
    #     'VEHICLE_CLASS',
    'FUEL_TYPE_ID',
    #   'CAR_STYLE_ID',
    'EMISSION_CLASS_ID',
    'DRIVE_ID',
    'MAKE_LABEL',
    'TRANSMISSION_ID',
    'SELLER_COUNTRY',
    'INTERIOR_MATERIAL'
]


def process_input(dict_input):
    initial_df = pd.DataFrame([dict_input])
    initial_df = pd.get_dummies(initial_df, columns=categorical_featues)

    # One-hot encoding for the 'MODEL' column
    one_hot_model = pd.get_dummies(initial_df['MODEL'], prefix='model')
    initial_df = pd.concat([initial_df, one_hot_model], axis=1)
    # initial_df = initial_df.drop(columns = ['MODEL'])

    # Convert the DataFrame to a dictionary
    df_dict = initial_df.iloc[0].to_dict()

    # Load model features
    with open('model/car_features.json', 'r') as f:
        model_features = json.load(f)

    # Add missing keys to df_dict and set them to 0
    for feature in model_features.keys():
        if feature not in df_dict:
            df_dict[feature] = 0

    # Create a new DataFrame based on the updated dictionary
    final_df = pd.DataFrame([df_dict])
    final_df = final_df[list(model_features.keys())]

    # Convert all columns to float
    final_df = final_df.astype(float)

    final_df.to_csv('mercedes_prediction.csv', index = None)

    # Return the final DataFrame
    return final_df