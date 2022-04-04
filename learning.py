import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from os import path
from tqdm import tqdm
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import joblib
import numpy as np
from torch.utils.data import DataLoader

from dataset_utils import SiamLikeDataset, train_val_test_split
from models import CombinedModel
from losses import ContrastiveLoss

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    current_dir = path.join(*path.split(Path(__file__).absolute())[:-1])
    data_dir = path.join(current_dir, 'data')
    markup = pd.read_csv(path.join(data_dir, 'markup.csv'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, valid, test = train_val_test_split(
        markup, test_size=0.25, valid_size=0.25, random_state=42, stratify='target'
    )
    dtst_train = SiamLikeDataset(markup=train,
                                 transactions_path=path.join(data_dir, 'transaction_data'),
                                 clickstream_path=path.join(data_dir, 'clickstream_data'))
    dtst_valid = SiamLikeDataset(markup=valid,
                                 transactions_path=path.join(data_dir, 'transaction_data'),
                                 clickstream_path=path.join(data_dir, 'clickstream_data'))
    dtst_test = SiamLikeDataset(markup=test,
                                transactions_path=path.join(data_dir, 'transaction_data'),
                                clickstream_path=path.join(data_dir, 'clickstream_data'))
    batch_size = 128
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_dataloader = DataLoader(dtst_train, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    valid_dataloader = DataLoader(dtst_valid, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_dataloader = DataLoader(dtst_test, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    le_mcc = joblib.load(path.join(data_dir, 'models_objects', 'le_mcc'))
    le_currency_rk = joblib.load(path.join(data_dir, 'models_objects', 'le_currency_rk'))
    le_click_categories = joblib.load(path.join(data_dir, 'models_objects', 'le_click_categories'))
    model = CombinedModel(mcc_classes=len(le_mcc.classes_),
                          mcc_emb_size=3,
                          currency_rk_classes=len(le_currency_rk.classes_),
                          currency_rk_emb_size=2, cat_id_classes=len(le_click_categories.classes_),
                          cat_id_emb_size=5, device='cpu').to(device)

    loss = ContrastiveLoss(1)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 50

    train_epoch_losses = []
    valid_epoch_losses = []
    es_counter = 0
    for epoch in range(n_epochs):
        with tqdm(train_dataloader, unit="batch") as tqdm_train_dataloader:
            model.train()
            train_loss = []
            for batch in tqdm_train_dataloader:
                tqdm_train_dataloader.set_description(f"train Epoch {epoch}")
                model.zero_grad()
                bank_out, rtk_out = model(batch)
                batch_loss = loss(bank_out, rtk_out, batch['target'], raw=True)
                batch_loss.mean().backward()
                optimizer.step()
                train_loss.extend(batch_loss.detach().tolist())
                tqdm_train_dataloader.set_postfix(
                    batch_loss=batch_loss.mean().item(),
                    epoch_loss=np.mean(train_loss) if not train_epoch_losses else np.mean(train_epoch_losses))
        with torch.no_grad():
            model.eval()
            with tqdm(valid_dataloader, unit="batch") as tqdm_valid_dataloader:
                valid_loss = []
                for batch in tqdm_train_dataloader:
                    tqdm_valid_dataloader.set_description(f"valid Epoch {epoch}")
                    bank_out, rtk_out = model(batch)
                    batch_loss = loss(bank_out, rtk_out, batch['target'], raw=True).detach()
                    valid_loss.extend(batch_loss.tolist())
                    tqdm_valid_dataloader.set_postfix(
                        batch_loss=batch_loss.mean().item(),
                        epoch_loss=np.mean(valid_loss) if not valid_epoch_losses else np.mean(valid_epoch_losses))
        train_epoch_losses.append(np.mean(train_loss))
        valid_epoch_losses.append(np.mean(valid_loss))
        now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        torch.save(model.state_dict(),
                   path.join(
                       data_dir,
                       'nn_chpt',
                       f'model_{now_time_str}_{round(train_epoch_losses[-1],5)}_{round(valid_epoch_losses[-1],5)}'))
        if (len(valid_epoch_losses) > 1) and (valid_epoch_losses[-1] >= min(valid_epoch_losses[:-1])):
            es_counter += 1
        elif (len(valid_epoch_losses) > 1) and (valid_epoch_losses[-1] < min(valid_epoch_losses[:-1])):
            es_counter = 0
        if es_counter == 5:
            break
