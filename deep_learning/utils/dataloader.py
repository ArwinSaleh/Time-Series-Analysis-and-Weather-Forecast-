from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Time Series dataset."""

    def __init__(self, csv_dir: Path, set_id: str = "train", transform=None):
        """
        Args:
            csv_dir (Path): Directory containing CSV files.
            set_id (str): Identifier for the dataset (e.g., "train", "test").
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.time_series_data_src = pd.read_csv(csv_dir / f"weather_data_{set_id}.csv")
        self.time_series_data_tgt = pd.read_csv(
            csv_dir / f"weather_data_{set_id}_labels.csv"
        )
        self.transform = transform

    def __len__(self):
        return len(
            self.time_series_data_src
        )  # Assuming both src and tgt have same length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        time_series_src = self.time_series_data_src.iloc[idx, 1:]
        time_series_tgt = self.time_series_data_tgt.iloc[idx, 1:]

        sample = {"source": time_series_src, "target": time_series_tgt}

        if self.transform:
            sample = self.transform(sample)

        return sample
