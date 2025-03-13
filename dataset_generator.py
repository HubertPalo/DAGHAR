from pathlib import Path
import os, shutil
import pandas as pd
from typing import Tuple, List, Dict
import argparse

from steps import (
    SplitGuaranteeingAllClassesPerSplit,
    BalanceToMinimumClass,
    BalanceToMinimumClassAndUser,
    FilterByCommonRows,
)

from pipelines import match_columns, pipelines

# Set the seed for reproducibility
import numpy as np
import random
import traceback
from itertools import product

np.random.seed(42)
random.seed(42)
# pd.np.random.seed(42)

from readers import (
    read_kuhar,
    read_motionsense,
    read_wisdm,
    read_uci,
    read_realworld,
    sanity_function,
    real_world_organize,
    read_recodgait_v2,
)

from concatenated_datasets_helper import create_dataset, apply_custom_functions_to_every_csv_file, set_gyroscope_values_to_zero, add_fake_activity_code_channel

"""This module is used to generate the datasets. The datasets are generated in the following steps:    
    1. Read the raw dataset
    2. Preprocess the raw dataset
    3. Preprocess the standardized dataset
    4. Remove activities that are equal to -1
    5. Balance the dataset per activity
    6. Balance the dataset per user and activity
    7. Save the datasets
    8. Generate the views of the datasets

    The datasets are generated in the following folders:
    1. data/unbalanced: The unbalanced dataset
    2. data/raw_balanced: The raw balanced dataset per activity
    3. data/standardized_balanced: The standardized balanced dataset per activity
    4. data/raw_balanced_user: The raw balanced dataset per user and activity (NOT USED)
    5. data/standardized_balanced_user: The standardized balanced dataset per user and activity (NOT USED)

    The datasets are generated in the following format:
    1. data/unbalanced/{dataset}/unbalanced.csv: The unbalanced dataset
    2. data/raw_balanced/{dataset}/train.csv: The raw balanced train dataset per activity
    3. data/raw_balanced/{dataset}/validation.csv: The raw balanced validation dataset per activity
    4. data/raw_balanced/{dataset}/test.csv: The raw balanced test dataset per activity
    5. data/standardized_balanced/{dataset}/train.csv: The standardized balanced train dataset per activity
    6. data/standardized_balanced/{dataset}/validation.csv: The standardized balanced validation dataset per activity
    7. data/standardized_balanced/{dataset}/test.csv: The standardized balanced test dataset per activity
    8. data/raw_balanced_user/{dataset}/train.csv: The raw balanced train dataset per user and activity (NOT USED)
    9. data/raw_balanced_user/{dataset}/validation.csv: The raw balanced validation dataset per user and activity (NOT USED)
    10. data/raw_balanced_user/{dataset}/test.csv: The raw balanced test dataset per user and activity (NOT USED)
    11. data/standardized_balanced_user/{dataset}/train.csv: The standardized balanced train dataset per user and activity (NOT USED)
    12. data/standardized_balanced_user/{dataset}/validation.csv: The standardized balanced validation dataset per user and activity (NOT USED)
    13. data/standardized_balanced_user/{dataset}/test.csv: The standardized balanced test dataset per user and activity (NOT USED)
"""

# Dictionary of dataset paths
dataset_paths: Dict[str, str] = {
    "KuHar": "KuHar/1.Raw_time_domian_data",
    "MotionSense": "MotionSense/A_DeviceMotion_data",
    "WISDM": "WISDM/wisdm-dataset/raw/phone",
    "UCI": "UCI/RawData",
    "RealWorld": "RealWorld/realworld2016_dataset",
    "HAPT": "UCI/RawData",
    "RecodGait_v2": "RecodGaitv2/RecodGait v2/raw_data",
}

# Dictionary with datasets and their respesctive reader functions
dataset_readers: Dict[str, callable] = {
    "KuHar": read_kuhar,
    "MotionSense": read_motionsense,
    "WISDM": read_wisdm,
    "UCI": read_uci,
    "RealWorld": read_realworld,
    "HAPT": read_uci,
    "RecodGait_v2": read_recodgait_v2,
}

# Preprocess the datasets

# Path to save the datasets
# output_path: Path = Path("data/datasets")

balancer_activity: object = BalanceToMinimumClass(
    class_column="standard activity code"
)
balancer_activity_and_user: object = BalanceToMinimumClassAndUser(
    class_column="standard activity code", filter_column="user"
)

split_data: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.8,
    random_state=42,
)

split_data_train_val: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.9,
    random_state=42,
)


def balance_per_activity(
    dataset: str, dataframe: pd.DataFrame, output_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function balance the dataset per activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The first element is the train dataset, the second is the validation dataset and the third is the test dataset.
    """

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity(train_df)
    val_df = balancer_activity(val_df)
    test_df = balancer_activity(test_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Raw data balanced per activity saved at {output_dir}")

    return train_df, val_df, test_df


def balance_per_user_and_activity(dataset, dataframe, output_path):
    """The function balance the dataset per user and activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The first element is the train dataset, the second is the validation dataset and the third is the test dataset.
    """
    new_df_balanced = balancer_activity_and_user(
        dataframe[dataframe["standard activity code"] != -1]
    )
    train_df, test_df = split_data(new_df_balanced)
    train_df, val_df = split_data_train_val(train_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Raw data balanced per user and activity saved at {output_dir}")

    return train_df, val_df, test_df


def generate_views(
    new_df,
    new_df_standardized,
    dataset,
    path_balanced,
    path_balanced_standardized,
):
    """This function generate the views of the dataset.

    Parameters
    ----------
    new_df : pd.DataFrame
        The raw dataset.
    new_df_standardized : pd.DataFrame
        The standardized dataset.
    dataset : str
        The dataset name.
    """

    # Filter the datasets by equal elements
    filter_common = FilterByCommonRows(match_columns=match_columns[dataset])
    new_df, new_df_standardized = filter_common(new_df, new_df_standardized)

    # Preprocess and save the raw balanced dataset per activity
    print(" ---- RAW")
    train_df, val_df, test_df = balance_per_activity(
        dataset, new_df, path_balanced
    )
    sanity_function(train_df, val_df, test_df)

    # Preprocess and save the standardized balanced dataset per activity
    print(" ---- STANDARDIZED")
    train_df, val_df, test_df = balance_per_activity(
        dataset, new_df_standardized, path_balanced_standardized
    )
    sanity_function(train_df, val_df, test_df)


def generate_hapt_views(data: pd.DataFrame, train_user_ids: List[int], val_user_ids: List[int], test_user_ids: List[int], output_path: Path):
    # Separate the data per users
    print(data.head())
    train_data = data[data['user'].isin(train_user_ids)].sort_values(['user', 'activity code', 'window']).reset_index(drop=True)
    val_data = data[data['user'].isin(val_user_ids)].sort_values(['user', 'activity code', 'window']).reset_index(drop=True)
    test_data = data[data['user'].isin(test_user_ids)].sort_values(['user', 'activity code', 'window']).reset_index(drop=True)
    # Saving the dataframes
    to_delete_output_path = output_path / "to_delete" / "HAPT"
    os.makedirs(to_delete_output_path, exist_ok=True)           
    train_data.to_csv(to_delete_output_path / "train.csv", index=False)
    val_data.to_csv(to_delete_output_path / 'validation.csv', index=False)
    test_data.to_csv(to_delete_output_path / 'test.csv', index=False)
    # For concatenated datasets
    concatenated_path = output_path / "HAPT" / "HAPT_concatenated_in_user_files"
    create_dataset(
        to_delete_output_path,
        concatenated_path,
        label=None,
        columns_to_maintain_in_linearize_dataframe=["user"],
        column_prefixes=["accel-x", "accel-y", "accel-z"]
    )

    balanced_data_per_user_and_class = []
    for user in data['user'].unique():
        balanced_data_per_user_and_class.append(BalanceToMinimumClass()(data[data['user'] == user].copy()))
    balanced_data_per_user_and_class = pd.concat(balanced_data_per_user_and_class)
    
    balanced_train_data = balanced_data_per_user_and_class[balanced_data_per_user_and_class['user'].isin(train_user_ids)]
    balanced_val_data = balanced_data_per_user_and_class[balanced_data_per_user_and_class['user'].isin(val_user_ids)]
    balanced_test_data = balanced_data_per_user_and_class[balanced_data_per_user_and_class['user'].isin(test_user_ids)]
    
    balanced_data_path = output_path / 'HAPT' / 'HAPT_daghar_like'
    balanced_data_path.mkdir(parents=True, exist_ok=True)

    balanced_train_data.to_csv(balanced_data_path / 'train.csv', index=False)
    balanced_val_data.to_csv(balanced_data_path / 'validation.csv', index=False)
    balanced_test_data.to_csv(balanced_data_path / 'test.csv', index=False)

def generate_rg_views(data: pd.DataFrame, train_user_ids: List[int], val_user_ids: List[int], test_user_ids: List[int], output_path: Path):
    # Separate the data per users
    train_data = data[data['user'].isin(train_user_ids)].sort_values(['user', 'session', 'window']).reset_index(drop=True)
    val_data = data[data['user'].isin(val_user_ids)].sort_values(['user', 'session', 'window']).reset_index(drop=True)
    test_data = data[data['user'].isin(test_user_ids)].sort_values(['user', 'session', 'window']).reset_index(drop=True)
    # Saving the dataframes
    to_delete_output_path = output_path / "to_delete" / "RecodGait_v2"
    os.makedirs(to_delete_output_path, exist_ok=True)           
    train_data.to_csv(to_delete_output_path / "train.csv", index=False)
    val_data.to_csv(to_delete_output_path / 'validation.csv', index=False)
    test_data.to_csv(to_delete_output_path / 'test.csv', index=False)
    # For concatenated datasets
    concatenated_temporary_path = output_path / "to_delete" / "RG_concatenated_in_user_files"
    create_dataset(
        to_delete_output_path,
        concatenated_temporary_path,
        label=None,
        columns_to_maintain_in_linearize_dataframe=["user"],
        column_prefixes=["accel-x", "accel-y", "accel-z"]
    )
    for partition in ['train', 'validation', 'test']:
        apply_custom_functions_to_every_csv_file(
            input_path=concatenated_temporary_path / partition,
            output_path=output_path / "RecodGait_v2" / "RG_concatenated_in_user_files" / partition,
            custom_functions=[set_gyroscope_values_to_zero, add_fake_activity_code_channel]
        )

    # Add gyroscope values if they are not present
    if 'gyro' not in data.columns:
        gyro_fake_df = pd.DataFrame(np.zeros((len(data), 180)), columns=[f'gyro-{axis}-{timestamp}' for axis in ['x', 'y', 'z'] for timestamp in range(60)])
        data = pd.concat([data, gyro_fake_df], axis=1)
    # Generate the index dictionary
    data_index_dict = {
        partition: {
            user_id: {
                session: [
                    int(val)
                    for val in data[(data['user'] == user_id)&(data['session'] == session)].index
                    ]
                for session in data[data['user'] == user_id]['session'].unique().tolist()
            }
            for user_id in user_ids
        }
        for partition, user_ids in zip(['train', 'validation', 'test'], [train_user_ids, val_user_ids, test_user_ids])
    }
    os.makedirs(output_path / "RecodGait_v2" / "RG_daghar_like", exist_ok=True)
    for partition in ['train', 'validation', 'test']:
        # Generate the index pairs dataframes
        index_pairs_df = generate_pairs_from_rg_partition(partition, data_index_dict)
        # Save the dataframes
        index_pairs_df.to_csv(output_path / "RecodGait_v2" / f"RG_{partition}_index_pairs.csv", index=False)
        mix_data_according_to_pairs(index_pairs_df, data, output_path / "RecodGait_v2" / "RG_daghar_like" / f"{partition}.csv")
    
    # train_index_pairs_df = generate_pairs_from_rg_partition('train', data_index_dict)
    # val_index_pairs_df = generate_pairs_from_rg_partition('validation', data_index_dict)
    # test_index_pairs_df = generate_pairs_from_rg_partition('test', data_index_dict)
    
    # train_index_pairs_df.to_csv('RG_train_index_pairs.csv', index=False)
    # val_index_pairs_df.to_csv('RG_val_index_pairs.csv', index=False)
    # test_index_pairs_df.to_csv('RG_test_index_pairs.csv', index=False)
    # mix_data_according_to_pairs(train_index_pairs_df, data, output_path / 'train.csv')
    # mix_data_according_to_pairs(val_index_pairs_df, data, output_path / 'validation.csv')
    # mix_data_according_to_pairs(test_index_pairs_df, data, output_path / 'test.csv')

# For RECODGAIT dataset
def generate_pairs_from_rg_partition(partition: str, data_dict: dict, positive_pairs_per_user: int=500):
    assert partition in ['train', 'validation', 'test'], 'Invalid partition'
    negative_pairs_per_user = positive_pairs_per_user * 5
    pairs = []
    partition_data_dict = data_dict[partition]
    users_ids = list(partition_data_dict.keys())
    for user_id in users_ids:
        unique_session_ids = list(partition_data_dict[user_id].keys())
        # Positive pairs
        for _ in range(positive_pairs_per_user):
            random_session_id_1, random_session_id_2 = random.sample(unique_session_ids, k=2)
            user_sample_index_1 = random.choice(partition_data_dict[user_id][random_session_id_1])
            user_sample_index_2 = random.choice(partition_data_dict[user_id][random_session_id_2])
            # Append the pair
            pairs.append({
                'sample-1-user-id': user_id,
                'sample-2-user-id': user_id,
                'sample-1-session-id': random_session_id_1,
                'sample-2-session-id': random_session_id_2,
                'sample-1-index': user_sample_index_1,
                'sample-2-index': user_sample_index_2,
                'label': 1
            })
        # Negative pairs
        other_users_ids = [other_user_id for other_user_id in users_ids if user_id != other_user_id]
        # print(f'User {user_id} - USERS {users_ids} - IMPOSTORS: {other_users_ids}')
        for _ in range(negative_pairs_per_user):
            # Original user sample selection
            user_session_id = 1 if partition == 'test' else random.choice(unique_session_ids)
            user_sample_index = random.choice(partition_data_dict[user_id][user_session_id])
            # Impostor user sample selection
            random_impostor_user_id = random.choice(list(other_users_ids))
            random_impostor_session_ids = list(partition_data_dict[random_impostor_user_id].keys())
            random_impostor_session_id = 3 if partition == 'test' else random.choice(random_impostor_session_ids)
            random_impostor_sample_index = random.choice(partition_data_dict[random_impostor_user_id][random_impostor_session_id])
            # Append the pair
            pairs.append({
                'sample-1-user-id': user_id,
                'sample-2-user-id': random_impostor_user_id,
                'sample-1-session-id': user_session_id,
                'sample-2-session-id': random_impostor_session_id,
                'sample-1-index': user_sample_index,
                'sample-2-index': random_impostor_sample_index,
                'label': 0
            })
    return pd.DataFrame(pairs)

# Mix the data based on defined pairs
def mix_data_according_to_pairs(pairs_data: pd.DataFrame, samples_data: pd.DataFrame, output_path: Path):
    sample1_indexes = pairs_data['sample-1-index'].tolist()
    sample2_indexes = pairs_data['sample-2-index'].tolist()
    
    preffixes = ['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']

    filter_columns = [f'{preffix}-{i}' for preffix, i in product(preffixes, range(60))]
    additional_columns = ['user', 'session']
    
    filter_columns = filter_columns + additional_columns

    sample1_data = samples_data.iloc[sample1_indexes][filter_columns]
    sample2_data = samples_data.iloc[sample2_indexes][filter_columns]

    # Rename columns
    sample1_data.columns = [f'sample-1-{col}' for col in sample1_data.columns]
    sample2_data.columns = [f'sample-2-{col}' for col in sample2_data.columns]
    # # Adding zero gyroscope values
    # for col in ['gyro-x', 'gyro-y', 'gyro-z']:
    #     for i in range(60):
    #         sample1_data[f'sample-1-{col}-{i}'] = 0.0
    #         sample2_data[f'sample-2-{col}-{i}'] = 0.0
    # Reset the indexes
    sample1_data.reset_index(drop=True, inplace=True)
    sample2_data.reset_index(drop=True, inplace=True)
    # Concatenate the data
    final_data = pd.concat([sample1_data, sample2_data, pairs_data], axis=1)
    final_data.to_csv(output_path, index=False)


def main(datasets_to_process: List[str], output_path: str):
    """This is the main function to generate the datasets. It will loop through
    and their respective pipelines to generate the datasets.

    Parameters
    ----------
    datasets_to_process : List[str]
        A list of datasets to process.
    output_path : str
        The path to save the datasets.
    """
    output_path = Path(output_path)
    # Creating the datasets
    for dataset in datasets_to_process:
        print(f"Preprocessing the dataset {dataset} ...\n")

        reader = dataset_readers[dataset]

        # Read the raw dataset
        if dataset == "RealWorld":
            print("Organizing the RealWorld dataset ...\n")
            # Create a folder to save the organized dataset
            workspace = Path(
                "data/original/RealWorld/realworld2016_dataset_organized"
            )
            if not os.path.isdir(workspace):
                os.mkdir(workspace)
            # Organize the dataset
            workspace, users = real_world_organize()
            path = workspace
            raw_dataset = reader(path, users)
            # Preprocess the raw dataset
            print(f"Preprocess the raw dataset: {dataset}\n")
            new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
            for view_name in pipelines[dataset].keys():
                try:
                    if "standardized" in view_name:
                        print(
                            f"Preprocess the dataset {dataset} with {view_name} ...\n"
                        )

                        # Preprocess the standardized dataset
                        new_df_standardized = pipelines[dataset][view_name](
                            raw_dataset
                        )
                        # Remove activities that are equal to -1
                        new_df = new_df[new_df["standard activity code"] != -1]
                        new_df_standardized = new_df_standardized[
                            new_df_standardized["standard activity code"] != -1
                        ]
                        generate_views(
                            new_df,
                            new_df_standardized,
                            dataset,
                            path_balanced=output_path / f"raw_balanced",
                            path_balanced_standardized=output_path / f"{view_name}_balanced",
                        )
                        positions = new_df["position"].unique()
                        for position in list(positions):
                            new_df_filtered = new_df[
                                new_df["position"] == position
                            ]
                            new_df_standardized_filtered = new_df_standardized[
                                new_df_standardized["position"] == position
                            ]
                            new_dataset = dataset + "_" + position
                            generate_views(
                                new_df_filtered,
                                new_df_standardized_filtered,
                                new_dataset,
                                path_balanced=output_path  / f"raw_balanced",
                                path_balanced_standardized=output_path / f"{view_name}_balanced",
                            )
                except Exception as e:
                    print(
                        f"Error generating the view {view_name} for {dataset}: {e}"
                    )
                    traceback.print_exc()
                    continue
        elif dataset == "RecodGait_v2":
            standardized_unbalanced_path = output_path / "standardized_unbalanced" / dataset
            print("Checking for RecodGait standardized unbalanced data...")
            if not os.path.exists(standardized_unbalanced_path / "standardized_unbalanced.csv"):
                print("Unbalanced data do not exist. Creating...")
                os.makedirs(standardized_unbalanced_path, exist_ok=True)
                path = Path(f"data/original/{dataset_paths[dataset]}")
                raw_dataset = reader(path)
                # Preprocess the dataset
                data = pipelines[dataset]["standardized"](raw_dataset)
                data.to_csv(standardized_unbalanced_path / "standardized_unbalanced.csv", index=False)
            else:
                print("Unbalanced data found. Reading...")
                data = pd.read_csv(standardized_unbalanced_path / "standardized_unbalanced.csv")
            # Preprocess the data
            data['user'] = data['user'].apply(lambda x: int(x))
            data['session'] = data['session'].apply(lambda x: int(x))
            # Group by user and create a column for every unique sessions
            data_sessions = data.groupby(['user', 'session']).size().unstack(fill_value=0)
            data_sessions['S-count'] = data_sessions.apply(lambda row: 5 - row.value_counts().get(0, 0), axis=1)
            # Separate the data in train, validation and test to identify the respective user ids
            train_user_ids = data_sessions[data_sessions['S-count'].isin([3,5])].index
            val_user_ids = data_sessions[data_sessions['S-count'] == 4].index
            test_user_ids = data_sessions[data_sessions['S-count'] == 2].index
            generate_rg_views(data, train_user_ids, val_user_ids, test_user_ids, output_path)
        elif dataset == "HAPT":
            standardized_unbalanced_path = output_path / "standardized_unbalanced" / dataset
            print("Checking for HAPT standardized unbalanced data...")
            if not os.path.exists(standardized_unbalanced_path / "standardized_unbalanced.csv"):
                print("Unbalanced data do not exist. Creating...")
                os.makedirs(standardized_unbalanced_path, exist_ok=True)
                path = Path(f"data/original/{dataset_paths[dataset]}")
                raw_dataset = reader(path)
                # Preprocess the dataset
                data = pipelines[dataset]["standardized"](raw_dataset)
                data.to_csv(standardized_unbalanced_path / "standardized_unbalanced.csv", index=False)
            else:
                print("Unbalanced data found. Reading...")
                data = pd.read_csv(standardized_unbalanced_path / "standardized_unbalanced.csv")
            # Preprocess the data
            data['user'] = data['user'].apply(lambda x: int(x))
            # Get the ids of the users in the train and test datasets
            users_in_train_df = pd.read_csv('data/original/UCI/Train/subject_id_train.txt', header=None)
            users_in_train_ids = [int(user_id) for user_id in users_in_train_df[0].unique()]
            users_in_test_df = pd.read_csv('data/original/UCI/Test/subject_id_test.txt', header=None)
            users_in_test_ids = [int(user_id) for user_id in users_in_test_df[0].unique()]

            train_data = data[data['user'].isin(users_in_train_ids)]
            grouped_train_data = train_data.groupby(['user', 'standard activity code']).count()[['activity code']].reset_index()
            grouped_train_data = grouped_train_data.groupby('user').min()['activity code'].sort_values().reset_index()
            users_in_val_ids = [int(user_id) for user_id in grouped_train_data.iloc[-4:]['user'].tolist()]

            generate_hapt_views(data, users_in_train_ids, users_in_val_ids, users_in_test_ids, output_path)
            # assert False

            
            # balanced_data_per_user_and_class = []
            # for user in data['user'].unique():
            #     print(f"User {user} - {len(data[data['user'] == user])}")
            #     balanced_data_per_user_and_class.append(BalanceToMinimumClass()(data[data['user'] == user].copy()))
            # balanced_data_per_user_and_class = pd.concat(balanced_data_per_user_and_class)
            # train_data = balanced_data_per_user_and_class[~balanced_data_per_user_and_class['user'].isin(users_in_val_ids)]
            # val_data = balanced_data_per_user_and_class[balanced_data_per_user_and_class['user'].isin(users_in_val_ids)]
            # test_data = data[data['user'].isin(users_in_test_ids)]
            
            # train_data.to_csv(output_path / 'train.csv', index=False)
            # val_data.to_csv(output_path / 'validation.csv', index=False)
            # test_data.to_csv(output_path / 'test.csv', index=False)
            # # display(train_data.groupby(['user', 'standard activity code']).count()[['activity code']])
            # # assert False
            # train_data.to_csv(output_path / 'train.csv', index=False)
            # val_data.to_csv(output_path / 'validation.csv', index=False)
            # test_data.to_csv(output_path / 'test.csv', index=False)
            # balanced_data_per_user_and_class = []
            # for user_id in users_in_train_ids:
            #     user_data = data[data['user'] == user_id]
            #     user_data_to_append = BalanceToMinimumClass()(user_data.copy())
            #     # if balance_function:
            #     #     user_data_to_append = balance_function(user_data_to_append)
            #     balanced_data_per_user_and_class.append(user_data_to_append)
            # balanced_train_data_per_user_and_class = pd.concat(balanced_data_per_user_and_class)

            # grouped_train_data = balanced_train_data_per_user_and_class.groupby(['user', 'standard activity code']).count()[['activity code']]
            # # print(grouped_train_data)
            # users_in_val_ids = grouped_train_data.pivot_table(index='user', columns='standard activity code', values='activity code', fill_value=0).sort_values(9)[-3:].index
            
            # balanced_data_per_user_and_class = []
            # for user_id in users_in_test_ids:
            #     user_data = data[data['user'] == user_id]
            #     balanced_data_per_user_and_class.append(BalanceToMinimumClass()(user_data.copy()))
            # balanced_test_data_per_user_and_class = pd.concat(balanced_data_per_user_and_class)
            
            # train_data = balanced_train_data_per_user_and_class[~balanced_train_data_per_user_and_class['user'].isin(users_in_val_ids)]
            # val_data = balanced_train_data_per_user_and_class[balanced_train_data_per_user_and_class['user'].isin(users_in_val_ids)]
            # test_data = balanced_test_data_per_user_and_class

            # train_data.to_csv(output_path / 'train.csv', index=False)
            # val_data.to_csv(output_path / 'validation.csv', index=False)
            # test_data.to_csv(output_path / 'test.csv', index=False)
        else:
            path = Path(f"data/original/{dataset_paths[dataset]}")
            raw_dataset = reader(path)
            # Preprocess the raw dataset
            print(f"Preprocess the raw dataset {dataset}\n")
            new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
            for view_name in pipelines[dataset].keys():
                try:
                    if "standardized" in view_name:
                        print(
                            f"Preprocess the dataset {dataset} with {view_name} ...\n"
                        )

                        # Preprocess the standardized dataset
                        new_df_standardized = pipelines[dataset][view_name](
                            raw_dataset
                        )
                        # Remove activities that are equal to -1
                        new_df = new_df[new_df["standard activity code"] != -1]
                        new_df_standardized = new_df_standardized[
                            new_df_standardized["standard activity code"] != -1
                        ]
                        
                        generate_views(
                            new_df,
                            new_df_standardized,
                            dataset,
                            path_balanced=output_path / f"raw_balanced",
                            path_balanced_standardized=output_path / f"{view_name}_balanced",
                        )
                except Exception as e:
                    print(
                        f"Error generating the view {view_name} for {dataset}: {e}"
                    )
                    traceback.print_exc()
                    continue

    # Remove the junk folder
    workspace = Path("data/processed")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    # Remove the realworld2016_dataset_organized folder
    workspace = Path("data/original/RealWorld/realworld2016_dataset_organized")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)


if __name__ == "__main__":    
    choices = [
        "KuHar",
        "MotionSense",
        "WISDM",
        "UCI",
        "RealWorld",
        "HAPT",
        "RecodGait_v2"
    ]
    
    parser = argparse.ArgumentParser(description="Dataset Generator")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to process. If not provided, all datasets will be processed.",
        choices=choices,
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the datasets",
        default="data/views",
        required=False,
    )
    
    args = parser.parse_args()
    datasets_to_process = choices
    if args.dataset:
        datasets_to_process = args.dataset

        
    print(f"Datasets to process: {datasets_to_process}")
    main(datasets_to_process, args.output_path)
