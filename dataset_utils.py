import pandas as pd
import numpy as np
from typing import Union, NoReturn, Iterable
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import joblib
from os import path


def train_val_test_split(data, test_size=None,
                         valid_size=None,
                         random_state=42,
                         stratify=None):
    train_valid, test = train_test_split(data, test_size=test_size, random_state=random_state,  shuffle=True,
                                         stratify=data[stratify] if stratify else None)
    train, valid = train_test_split(train_valid, test_size=valid_size/(1-test_size),
                                    random_state=random_state, shuffle=True,
                                    stratify=train_valid[stratify] if stratify else None)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, valid, test


def pad_1d(array: Iterable, new_size):
    old_size = len(array)
    if old_size < new_size:
        return np.pad(array, pad_width=(0, new_size-old_size), mode='constant')
    elif old_size > new_size:
        return array[:new_size]
    else:
        return array


class TargetBuilder(object):
    """
    Класс формирования таргета с парами bank_id, rtk_id для дальнейшего использования.
    """
    def __init__(self, matching: Union[pd.DataFrame, str], random_state: int = 42) -> NoReturn:
        """
        :param matching: файл от организаторов с известными парами bank_id-rtk_id.
        :param random_state:
        """
        if isinstance(matching, pd.DataFrame):
            self.matching = matching.copy()
        else:
            self.matching = pd.read_csv(matching)
        self.rnd = check_random_state(random_state)

    def get_markup_df(self, target_rate: float = 0.1) -> pd.DataFrame:
        """
        Функция для создания разметки.
        :param target_rate: желаемый уровень целевого события.
        :return: датафрейм с разметкой вида bank_id-rtk_id-target
        """
        if (target_rate <= 0) or (target_rate >= 1):
            raise ValueError('Relevant target rate range is (0;1)')
        # Знаем, что 1 rtk соответствует 1 bank и наоборот
        n_positive = (self.matching.rtk != '0').sum()
        required_negative = np.ceil(n_positive / target_rate) - n_positive
        negative_pairs = []
        not_paired_banks = (self.matching.rtk == '0').sum()
        pairs_for_not_paired_bank = (self.matching.rtk != '0').sum()
        pairs_for_paired_bank = pairs_for_not_paired_bank - 1
        all_pairs = pairs_for_paired_bank ** 2 + not_paired_banks * (pairs_for_paired_bank + 1)
        self.matching.loc[self.matching.rtk != '0', 'pairs'] = pairs_for_paired_bank
        self.matching.loc[self.matching.rtk == '0', 'pairs'] = pairs_for_not_paired_bank
        self.matching.loc[:, 'pairs'] = self.matching.pairs.cumsum()
        for idx in tqdm(self.rnd.choice(all_pairs, size=int(required_negative), replace=False)):
            row = self.matching[self.matching.pairs > idx].head(1)
            negative_pairs.append(
                list(product(
                    [row['bank'].values[0]],
                    self.matching.rtk[~self.matching.rtk.isin(['0', row['rtk'].values[0]])].values
                ))[int(idx - row['pairs'].values[0])]
            )
        pos_df = self.matching.loc[self.matching.rtk != '0', ['bank', 'rtk']]
        neg_df = pd.DataFrame(negative_pairs, columns=['bank', 'rtk'])
        pos_df['target'] = 1
        neg_df['target'] = 0
        return pd.concat([pos_df, neg_df]).reset_index(drop=True)


class SiamLikeDataset(Dataset):

    def __init__(self, markup: Union[pd.DataFrame, str],
                 transactions_path: str,
                 clickstream_path: str,
                 bank_size: int = 1023,
                 rtk_size: int = 7211):
        if isinstance(markup, pd.DataFrame):
            self.markup_df = markup.copy()
        else:
            self.markup_df = pd.read_csv(markup)
        self.transactions_path = transactions_path
        self.clickstream_path = clickstream_path
        self.bank_size = bank_size
        self.rtk_size = rtk_size

    def __len__(self):
        return len(self.markup_df)

    def __getitem__(self, idx):
        row = self.markup_df.loc[idx, :]
        bank_id = row.bank
        rtk_id = row.rtk
        bank_data = joblib.load(path.join(self.transactions_path, bank_id))
        rtk_data = joblib.load(path.join(self.clickstream_path, rtk_id))

        assert bank_id == bank_data['user_id']
        assert rtk_id == rtk_data['user_id']

        observation_data = dict()
        observation_data['bank_id'] = bank_id
        observation_data['rtk_id'] = rtk_id

        observation_data['mcc_code'] = torch.from_numpy(pad_1d(bank_data['mcc_code'], self.bank_size))
        observation_data['currency_rk'] = torch.from_numpy(pad_1d(bank_data['currency_rk'], self.bank_size))
        observation_data['transaction_amt'] = torch.from_numpy(pad_1d(bank_data['transaction_amt'], self.bank_size))
        observation_data['cat_id'] = torch.from_numpy(pad_1d(rtk_data['cat_id'], self.rtk_size))
        observation_data['target'] = row.target
        return observation_data
