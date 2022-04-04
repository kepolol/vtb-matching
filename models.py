import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int = 3):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim, padding_idx=0),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        output = self.emb(x)
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, ):
        super().__init__()
        self.lstm_1d = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Dropout(p=0.1),
            nn.LSTM(input_size=input_size, hidden_size=64,
                    num_layers=1, batch_first=True,
                    dropout=0.1, bidirectional=True)
        )
        self.lstm_2d = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=64,
                    num_layers=1, batch_first=True, dropout=0.1, bidirectional=True)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            output, _ = self.lstm_1d(x)
        else:
            output, _ = self.lstm_2d(x)
        return torch.cat((output[:, -1, :64], output[:, 0, 64:]), dim=1)  # актуально для bidirectional


class BankModel(nn.Module):

    def __init__(self,
                 mcc_classes: int, mcc_emb_size: int,
                 currency_rk_classes: int, currency_rk_emb_size: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.emb_mcc = EmbeddingModel(num_embeddings=mcc_classes + 1,
                                      embedding_dim=mcc_emb_size)
        self.emb_currency_rk = EmbeddingModel(num_embeddings=currency_rk_classes + 1,
                                              embedding_dim=currency_rk_emb_size)
        self.lstm_mcc = LSTMModel(mcc_emb_size)
        self.lstm_currency_rk = LSTMModel(currency_rk_emb_size)
        self.lstm_transaction_amt = LSTMModel(1)

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        mcc_out = self.emb_mcc(x['mcc_code'].to(self.device))
        mcc_out = self.lstm_mcc(mcc_out)

        currency_rk_out = self.emb_mcc(x['currency_rk'].to(self.device))
        currency_rk_out = self.lstm_mcc(currency_rk_out)

        transaction_amt_out = self.lstm_transaction_amt(torch.unsqueeze(x['transaction_amt'].to(self.device).float(),
                                                                        2))

        out = torch.cat((mcc_out, currency_rk_out, transaction_amt_out), dim=1)
        out = self.fc(out)
        return out


class RTKModel(nn.Module):

    def __init__(self,
                 cat_id_classes: int, cat_id_emb_size: int, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.emb_cat_id = EmbeddingModel(num_embeddings=cat_id_classes + 1,
                                         embedding_dim=cat_id_emb_size)
        self.lstm_cat_id = LSTMModel(cat_id_emb_size)

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        cat_id_out = self.emb_cat_id(x['cat_id'].to(self.device))
        cat_id_out = self.lstm_cat_id(cat_id_out)
        out = self.fc(cat_id_out)
        return out


class CombinedModel(nn.Module):
    def __init__(self,
                 mcc_classes: int, mcc_emb_size: int,
                 currency_rk_classes: int, currency_rk_emb_size: int,
                 cat_id_classes: int, cat_id_emb_size: int, device: str = 'cpu'):
        super().__init__()
        self.m_bank = BankModel(mcc_classes=mcc_classes,
                                mcc_emb_size=mcc_emb_size,
                                currency_rk_classes=currency_rk_classes,
                                currency_rk_emb_size=currency_rk_emb_size, device=device)
        self.m_rtk = RTKModel(cat_id_classes=cat_id_classes,
                              cat_id_emb_size=cat_id_emb_size, device=device)

    def forward(self, x):
        bank_out = self.m_bank(x)
        rtk_out = self.m_rtk(x)
        return bank_out, rtk_out
