import pandas as pd
import re
from pathlib import Path
from typing import List, Callable


def apply_custom_functions_to_every_csv_file(input_path: Path, output_path: Path, custom_functions: List[Callable]):
    output_path.mkdir(parents=True, exist_ok=True)
    for file in input_path.glob("*.csv"):
        df = pd.read_csv(file)
        for custom_function in custom_functions:
            df = custom_function(df)
        df.to_csv(output_path / file.name, index=False)

def duplicate_accel_channels(df):
    # preffixes = ['accel-x', 'accel-y', 'accel-z']
    df['gyro-x'] = df['accel-x']
    df['gyro-y'] = df['accel-y']
    df['gyro-z'] = df['accel-z']
    return df

def set_gyroscope_values_to_zero(df):
    df["gyro-x"] = 0.0
    df["gyro-y"] = 0.0
    df["gyro-z"] = 0.0
    return df

def add_fake_activity_code_channel(df):
    df['activity code'] = 0
    return df

def linearize_dataframe(df: pd.DataFrame, column_prefixes, maintain):
    # Initialize a dictionary to hold the linearized columns
    linearized_data = {prefix: [] for prefix in column_prefixes}
    for m in maintain:
        linearized_data[m] = []

    # Regular expression to match columns that start with the prefix and are followed by a number
    def get_columns_with_number(df, prefix):
        return [col for col in df.columns if re.match(f"^{prefix}-\d+$", col)]

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # For each prefix, find how many columns correspond to it (prefix-N, where N is a number)
        for prefix in column_prefixes:
            cols = get_columns_with_number(df, prefix)
            # Extend the list with the corresponding values from the row
            linearized_data[prefix].extend([row[col] for col in cols])
        # Replicate the user column values for the number of new rows
        for m in maintain:
            linearized_data[m].extend([row[m]] * len(cols))

    # Create the new linearized DataFrame
    linearized_df = pd.DataFrame(linearized_data)

    # Return the linearized DataFrame
    return linearized_df


def create_dataset(
        root_path: Path,
        output_path: Path,
        label: str = "standard activity code",
        columns_to_maintain_in_linearize_dataframe=["user", "standard activity code"],
        column_prefixes = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    ):
    train_df = pd.read_csv(root_path / "train.csv")
    val_df = pd.read_csv(root_path / "validation.csv")
    test_df = pd.read_csv(root_path / "test.csv")

    # Linearize the dataframes
    print("Linearizing train dataframe...")
    train_df = linearize_dataframe(train_df, column_prefixes, columns_to_maintain_in_linearize_dataframe)
    print("Linearizing validation dataframe...")
    val_df = linearize_dataframe(val_df, column_prefixes, columns_to_maintain_in_linearize_dataframe)
    print("Linearizing test dataframe...")
    test_df = linearize_dataframe(test_df, column_prefixes, columns_to_maintain_in_linearize_dataframe)
    if label:
        train_df["activity code"] = train_df[label]
        val_df["activity code"] = val_df[label]
        test_df["activity code"] = test_df[label]
    
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        output_path_split = output_path / split
        output_path_split.mkdir(parents=True, exist_ok=True)
        for user_id, user_df in df.groupby("user"):
            user_df.to_csv(output_path_split / f"{user_id}.csv", index=False)
        print(f"Saved {len(df)} samples to {output_path}")
    
    return train_df, val_df, test_df